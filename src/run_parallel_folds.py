import argparse
import subprocess
import os
import sys
import time
import glob
import pandas as pd
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
import torch

def bootstrap_ci(df_oof):
    """
    Bootstrap the pooled OOF predictions to get Mean & CI.
    df_oof: DataFrame with "y_true" and "y_pred"
    """
    y_true_all = df_oof["y_true"].values
    y_pred_all = df_oof["y_pred"].values
    
    n_samples = len(y_true_all)
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    
    metrics = {"f1_macro": [], "accuracy": [], "kappa": []}
    
    print(f"Bootstrapping {n_bootstraps} times on {n_samples} samples...")
    for i in range(n_bootstraps):
        # Resample indices
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true_all[indices])) < 2:
            # Skip if only one class is present in bootstrap sample (can happen with small N)
            continue
            
        y_t = y_true_all[indices]
        y_p = y_pred_all[indices]
        
        metrics["f1_macro"].append(f1_score(y_t, y_p, average="macro"))
        metrics["accuracy"].append(accuracy_score(y_t, y_p))
        metrics["kappa"].append(cohen_kappa_score(y_t, y_p))
        
    stats = {}
    for key, values in metrics.items():
        if not values: continue
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        ci_lower = np.percentile(values, 2.5)
        ci_upper = np.percentile(values, 97.5)
        stats[key] = {
            "mean": mean_val,
            "std": std_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "str_ci": f"{mean_val:.4f} ({ci_lower:.4f}-{ci_upper:.4f})",
            "str_std": f"{mean_val:.4f} ± {std_val:.4f}"
        }
        
    return stats

def main():
    parser = argparse.ArgumentParser(description="Run parallel training repeats (Nested CV) for one or all folds.")
    
    # Arguments we explicitly handle/modify
    parser.add_argument("--fold-index", type=int, default=-1, help="Fold index to run. -1 to run ALL folds sequentially.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds (default 5).")
    parser.add_argument("--data-seed", type=int, default=42, help="Fixed seed for Outer Split (Data)")
    parser.add_argument("--log-dir", type=str, default="runs_parallel") # output dir
    parser.add_argument("--keep-checkpoints", action="store_true", help="If set, don't delete checkpoints after aggregation.")
    parser.add_argument("--jobs-per-gpu", type=int, default=1, help="Number of concurrent jobs per GPU (default 1).")
    
    # Capture everything else to pass to run.py
    args, unknown = parser.parse_known_args()
    
    # Identify model name
    model_name = "Model"
    if "--model" in unknown:
        try:
            idx = unknown.index("--model")
            model_name = unknown[idx+1]
        except: pass

    # Determine which folds to run
    if args.fold_index == -1:
        folds_to_run = range(args.n_splits)
        print(f"Running ALL {args.n_splits} folds sequentially.")
    else:
        folds_to_run = [args.fold_index]
        print(f"Running single fold: {args.fold_index}")

    all_folds_oof_dfs = []

    for fold_idx in folds_to_run:
        print(f" PROCESSING FOLD {fold_idx} / {args.n_splits - 1}")
        print(f"========================================")

        # Seeds for the 5 parallel runs (Inner Split & Model Init)
        seeds = [args.data_seed + i + 1 for i in range(5)]
        base_name = f"{model_name}_Fold{fold_idx}_Parallel"

        # --- 1. Prepare all tasks for this fold ---
        tasks = []
        for i, seed in enumerate(seeds):
            run_name = f"{model_name}_Fold{fold_idx}_Parallel_Rep{i}_Seed{seed}"
            tasks.append({
                "run_name": run_name,
                "seed": seed,
                "rep_idx": i
            })

        # --- 2. Determine Concurrency ---
        num_gpus = torch.cuda.device_count()
        # You have 24GB RAM, so we can safely run multiple jobs per GPU
        jobs_per_gpu = args.jobs_per_gpu
        max_parallel = max(1, num_gpus * jobs_per_gpu)
        
        print(f"  Detected {num_gpus} GPUs. Max parallel jobs: {max_parallel} ({jobs_per_gpu} per GPU)")

        active_processes = []
        task_idx = 0
        
        while task_idx < len(tasks) or active_processes:
            # Fill up slots
            while len(active_processes) < max_parallel and task_idx < len(tasks):
                t = tasks[task_idx]
                run_dir = os.path.join(args.log_dir, t["run_name"])
                os.makedirs(run_dir, exist_ok=True)
                log_path = os.path.join(run_dir, "process_output.log")
                
                # Simple GPU allocation: (task_idx % num_gpus)
                gpu_id = task_idx % num_gpus if num_gpus > 0 else -1
                
                env = os.environ.copy()
                if gpu_id != -1:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    print(f"  [Rep {t['rep_idx']}] Launching on GPU {gpu_id}...")
                else:
                    print(f"  [Rep {t['rep_idx']}] Launching on CPU...")

                cmd = [
                    sys.executable, "src/run.py",
                    "--fold-index", str(fold_idx),
                    "--data-seed", str(args.data_seed),
                    "--seed", str(t["seed"]),
                    "--name", t["run_name"],
                    "--log-dir", args.log_dir
                ]
                cmd.extend(unknown)
                
                log_file = open(log_path, "w")
                p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
                active_processes.append({"p": p, "log_f": log_file, "run_name": t["run_name"]})
                task_idx += 1

            # Check for finished processes
            for ap in active_processes[:]:
                if ap["p"].poll() is not None: # Process finished
                    exit_code = ap["p"].returncode
                    ap["log_f"].close()
                    if exit_code != 0:
                        print(f"  !! Error: {ap['run_name']} failed (code {exit_code}).")
                        # Print snippet of log
                        try:
                            # Use absolute path for log file safety
                            log_file_path = os.path.join(os.getcwd(), ap["log_f"].name)
                            with open(log_file_path, "r") as f:
                                lines = f.readlines()
                                snippet = "".join(lines[-15:]) # Get last 15 lines
                                print(f"--- LOG SNIPPET ({ap['run_name']}) ---\n{snippet}\n-------------------------")
                        except Exception as e:
                            print(f"Could not read log: {e}")
                    else:
                        print(f"  √ Finished: {ap['run_name']}")
                    active_processes.remove(ap)
            
            # --- 3. Progress Reporting ---
            if active_processes:
                prog_info = []
                for ap in active_processes:
                    try:
                        log_file_path = os.path.join(os.getcwd(), ap["log_f"].name)
                        with open(log_file_path, "r") as f:
                            content = f.read()
                            import re
                            # Find Epoch and Percentage: e.g. "Epoch 5:  80%|"
                            epoch_match = re.findall(r"Epoch\s+(\d+)", content)
                            pct_match = re.findall(r"(\d+)%\|", content)
                            
                            last_epoch = epoch_match[-1] if epoch_match else "0"
                            last_pct = int(pct_match[-1]) if pct_match else 0
                            
                            # Create a mini bar (length 10)
                            bar_len = 10
                            filled = int(last_pct / 100 * bar_len)
                            bar = "#" * filled + "-" * (bar_len - filled)
                            
                            rep_str = ap['run_name'].split('_')[-2].replace("Rep", "R") # R0, R1...
                            prog_info.append(f"{rep_str}: E{last_epoch} [{bar}] {last_pct}%")
                    except:
                        prog_info.append(f"R?: ?")
                
                # Update line (using a slightly safer way for Windows terminal)
                status_line = f"\r  Status: {' | '.join(prog_info)}"
                # Fill with spaces to clear old longer lines
                sys.stdout.write(status_line.ljust(120))
                sys.stdout.flush()



            time.sleep(5) # Check every 5 seconds
        print() # New line after tasks finished




        # --- Aggregate THIS Fold ---
        print(f"  Aggregating Fold {fold_idx} results...")
        fold_oof_dfs = []
        for i, seed in enumerate(seeds):
            run_name = f"{base_name}_Rep{i}_Seed{seed}"
            search_path = os.path.join(args.log_dir, run_name, "*_oof_predictions.csv")
            files = glob.glob(search_path)
            if files:
                df = pd.read_csv(files[0])
                df["repeat_seed"] = seed
                df["fold"] = fold_idx
                fold_oof_dfs.append(df)
        
        if fold_oof_dfs:
            df_fold_oof = pd.concat(fold_oof_dfs, ignore_index=True)
            
            # Save Fold Pooled OOF
            agg_dir = os.path.join(args.log_dir, base_name + "_Aggregated")
            os.makedirs(agg_dir, exist_ok=True)
            pooled_path = os.path.join(agg_dir, "pooled_oof_predictions.csv")
            df_fold_oof.to_csv(pooled_path, index=False)
            
            # Add to global list
            all_folds_oof_dfs.append(df_fold_oof)

            # --- Cleanup Checkpoints (Optional) ---
            if not args.keep_checkpoints:
                print(f"  Cleaning up checkpoints for Fold {fold_idx} to save space...")
                for i, seed in enumerate(seeds):
                    run_name = f"{base_name}_Rep{i}_Seed{seed}"
                    # Checkpoints are usually in log_dir/run_name/fold_X/checkpoints
                    # We can use glob to find and remove them
                    ckpt_pattern = os.path.join(args.log_dir, run_name, "fold_*", "checkpoints")
                    for ckpt_dir in glob.glob(ckpt_pattern):
                        try:
                            import shutil
                            shutil.rmtree(ckpt_dir)
                            # print(f"    Removed: {ckpt_dir}")
                        except Exception as e:
                            print(f"    Failed to remove {ckpt_dir}: {e}")
        else:
            print(f"  No OOF results found for Fold {fold_idx}")

    # --- Global Aggregation ---
    if all_folds_oof_dfs:
        print(f"\n========================================")
        print(f" FINAL GLOBAL AGGREGATION")
        print(f"========================================")
        
        df_global = pd.concat(all_folds_oof_dfs, ignore_index=True)
        
        global_agg_dir = os.path.join(args.log_dir, f"{model_name}_Global_Aggregated")
        os.makedirs(global_agg_dir, exist_ok=True)
        
        global_path = os.path.join(global_agg_dir, "global_pooled_oof_predictions.csv")
        df_global.to_csv(global_path, index=False)
        print(f"Saved Global OOF to: {global_path}")
        
        print("Bootstrapping Global Metrics...")
        stats = bootstrap_ci(df_global)
        
        print(json.dumps(stats, indent=2))
        with open(os.path.join(global_agg_dir, "global_bootstrapped_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
            
    print("\nDone.")

if __name__ == "__main__":
    main()
