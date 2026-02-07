
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

# ------------------------------------------------------------------------------
# Add src to sys.path
# ------------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# Also add parent dir for run.py to be importable if needed
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Imports moved to main() to avoid circular dependency


from model.resnet34_coralLoss import ResNet1D_CoralLoss
from data_provider.datamodule import TSDataModule

def get_args():
    parser = argparse.ArgumentParser(description="Optimize threshold for Coral Loss")
    
    # Checkpoint and Data
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the checkpoint file (.ckpt)")
    parser.add_argument("--root-dir", type=str, required=True, help="Path to the dataset CSV/folder")
    parser.add_argument("--dataset-name", type=str, default="data_300s_order5", help="Dataset name")

    # Split config to reproduce validation set
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--fold-index", type=int, default=0, help="Fold index to use for validation (0-indexed)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Data Loading params details
    parser.add_argument("--use-bmi", action="store_true", help="Include BMI features")
    parser.add_argument("--use-sex", action="store_true", help="Include sex features")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()

@torch.no_grad()
def evaluate_thresholds(model, data_loader, device, verbose=True):
    model.eval()
    model.to(device)
    
    all_logits = []
    all_targets = []
    
    if verbose:
        print(f"Running inference on validation set...")
    for batch in data_loader:
        x, y = batch
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())
            
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    y_true = all_targets.numpy()

    # Iterate thresholds
    thresholds = np.arange(0.1, 1.0, 0.05)
    
    results = []
    if verbose:
        print(f"\nEvaluating thresholds on {len(y_true)} samples...")
        print(f"{'Threshold':<10} {'Kappa':<10} {'F1-Macro':<10} {'Accuracy':<10}")
        print("-" * 45)

    best_kappa = -1.0
    best_thresh = 0.5

    for thresh in thresholds:
        # Decode logic: prob > thresh => sum
        probs = torch.sigmoid(all_logits)
        y_pred = (probs >= thresh).sum(dim=1).to(torch.int64).numpy()

        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        f1 = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)

        results.append({
            "threshold": thresh,
            "kappa": kappa,
            "f1": f1,
            "accuracy": acc
        })
        
        if verbose:
            print(f"{thresh:<10.2f} {kappa:<10.4f} {f1:<10.4f} {acc:<10.4f}")

        if kappa > best_kappa:
            best_kappa = kappa
            best_thresh = thresh

    if verbose:
        print("-" * 45)
        print(f"Best Threshold (Maximize Kappa): {best_thresh:.2f} -> Kappa: {best_kappa:.4f}")
    return results, best_thresh

def find_optimal_threshold(model, data_loader, device):
    """
    Wrapper for use in run.py
    """
    _, best_thresh = evaluate_thresholds(model, data_loader, device, verbose=False)
    return best_thresh

def main():
    args = get_args()
    print(f"Args: {args}")

    try:
        from run import load_dataset
    except ImportError:
        try:
            from src.run import load_dataset
        except ImportError:
             # Fallback if run is main
             import run
             load_dataset = run.load_dataset


    # 1. Load Data (Load all first to split)
    print(f"Loading full dataset from {args.root_dir}...")
    
    # Mocking args object for load_dataset
    class DataArgs:
        pass
    data_args = DataArgs()
    data_args.root_dir = args.root_dir
    data_args.dataset_name = args.dataset_name
    data_args.seed = args.seed
    data_args.use_bmi = args.use_bmi
    data_args.use_sex = args.use_sex
    
    data_ndarray = load_dataset(data_args)
    x_full = data_ndarray["x_full"]
    y_full = data_ndarray["y_full"]
    id_groups = data_ndarray["id_groups"]

    # 2. Recreate Split to get Valid set
    # Using same logic as run.py
    splitter_train_test = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    split_iter = list(splitter_train_test.split(x_full, y_full, id_groups))

    if not (0 <= args.fold_index < args.n_splits):
        raise ValueError(f"Fold index must be between 0 and {args.n_splits-1}")

    # Outer split (Train+Valid vs Test)
    train_valid_idx, test_idx = split_iter[args.fold_index]
    
    x_tv = x_full[train_valid_idx]
    y_tv = y_full[train_valid_idx]
    groups_tv = np.array(id_groups)[train_valid_idx]

    # Inner split (Train vs Valid)
    # run.py uses 5 splits for inner CV
    splitter_train_valid = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    inner_train_idx, inner_valid_idx = next(splitter_train_valid.split(x_tv, y_tv, groups_tv))
    
    # Map back to original indices if needed, but we just need the data subsets
    x_train = x_tv[inner_train_idx]
    y_train = y_tv[inner_train_idx]
    
    x_val = x_tv[inner_valid_idx]
    y_val = y_tv[inner_valid_idx]

    print(f"Data shapes: Train={x_train.shape}, Val={x_val.shape}")

    # 3. Setup DataModule (for standardization and loading)
    # We use x_val for both val and test slot to keep it simple
    dm = TSDataModule(
        x_train, y_train, x_val, y_val, x_val, y_val,
        standardize=True,  # Assuming we always want standardization as per run.py defaults
        use_weighted_sampler=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    dm.setup()
    val_loader = dm.val_dataloader()

    # 4. Load Model
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    # We need to map location if no cuda
    map_location = torch.device(args.device)
    model = ResNet1D_CoralLoss.load_from_checkpoint(args.checkpoint_path, map_location=map_location)
    
    # 5. Optimize
    evaluate_thresholds(model, val_loader, args.device)

if __name__ == "__main__":
    main()
