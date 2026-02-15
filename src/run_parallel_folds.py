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
        print(f"\n========================================")
        print(f" PROCESSING FOLD {fold_idx} / {args.n_splits - 1}")
        print(f"========================================")

        # Seeds for the 5 parallel runs (Inner Split & Model Init)
        seeds = [args.data_seed + i + 1 for i in range(5)]
        
        processes = []
        base_name = f"{model_name}_Fold{fold_idx}_Parallel"
        
        print(f"--- Launching 5 Parallel Repeats for Fold {fold_idx} ---")
        
        for i, seed in enumerate(seeds):
            run_name = f"{base_name}_Rep{i}_Seed{seed}"
            
            # Construct command for run.py
            cmd = [
                sys.executable, "src/run.py",
                "--fold-index", str(fold_idx),
                "--data-seed", str(args.data_seed),
                "--seed", str(seed),
                "--name", run_name,
                "--log-dir", args.log_dir
            ]
            cmd.extend(unknown)
            
            # print(f"  [Rep {i}] Launching...")
            p = subprocess.Popen(cmd)
            processes.append(p)
            
        print("  Waiting for 5 repeats to finish...")
        exit_codes = [p.wait() for p in processes]
        
        if any(c != 0 for c in exit_codes):
            print(f"  Warning: Some repeats failed for Fold {fold_idx}.")

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
            
            # Optional: Calc stats for just this fold
            # stats = bootstrap_ci(df_fold_oof)
            # with open(os.path.join(agg_dir, "bootstrapped_stats.json"), "w") as f:
            #     json.dump(stats, f, indent=2)
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
