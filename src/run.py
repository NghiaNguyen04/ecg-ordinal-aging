import warnings
# from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning
# warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)


import os, re
from pathlib import Path
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from model.resnet34_hybridLoss import ResNet1DLightning
from model.resnet34_coralLoss import ResNet1D_CoralLoss
from model.resnet34_FocalCosLoss import ResNet1D_FocalCos
from model.optimize_threshold import find_optimal_threshold


from data_provider.datamodule import TSDataModule
from data_provider.data_loader import AAGINGLoader
from data_provider.oversampling import ADASYNWrapper

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix

torch.set_float32_matmul_precision('high')

# ------------------------------- utilities -------------------------------

# def coral_pos_weight_from_labels(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
#     """
#     y_train: (N,), integer [0..K-1]
#     trả về tensor (K-1,) trên CPU (.to(device) sau)
#     pos_weight_k = N_neg_k / N_pos_k với nhãn nhị phân 1[y>k]
#     """
#     K = num_classes
#     weights = []
#     for k in range(K-1):
#         pos = (y_train > k).sum()
#         neg = (y_train <= k).sum()
#         # tránh chia 0
#         pos = max(pos, 1)
#         neg = max(neg, 1)
#         weights.append(neg / pos)
#     return torch.tensor(weights, dtype=torch.float32)

def load_dataset(args: argparse.Namespace):
    if args.dataset_name == "tfresh":
        from data_provider.data_loader import TsfreshLoader
        # tfresh vẫn dùng root_dir làm thư mục chứa nhiều file
        return TsfreshLoader(args.root_dir, seed=args.seed).load()
    
    # Mặc định sử dụng AAGINGLoader, truyền thẳng đường dẫn file CSV từ root_dir
    loader = AAGINGLoader(
        csv_path=args.root_dir, 
        seed=args.seed, 
        use_bmi=args.use_bmi, 
        use_sex=args.use_sex
    )
    return loader.load()

def load_model(args: argparse.Namespace, class_weights, y_train_np: np.ndarray = None) -> pl.LightningModule:
    if args.model == "Resnet34_hybrid":
        return ResNet1DLightning(
            in_channels=args.in_channels,
            nb_classes=args.nb_classes,
            lr=args.lr,
            class_weights=class_weights,
            sklearn_average=(None if args.average == "none" else args.average),
            use_bmi = args.use_bmi,
            use_sex = args.use_sex,
        )
    elif args.model == "Resnet34_coral":
        print("Using CoralLoss")
        # pw = coral_pos_weight_from_labels(y_train_np, num_classes=args.nb_classes)
        return ResNet1D_CoralLoss(
            in_channels=args.in_channels,
            nb_classes=args.nb_classes,
            lr=args.lr,
            sklearn_average=(None if args.average == "none" else args.average),
            # pos_weight=pw.to(args.device),
            pos_weight=None,
            threshold=args.threshold,
        )
    elif args.model == "Resnet34_FocalCos":
        return ResNet1D_FocalCos(
            in_channels=args.in_channels,
            nb_classes=args.nb_classes,
            lr=args.lr,
            class_weights=class_weights,
            sklearn_average=(None if args.average == "none" else args.average),
            use_bmi=args.use_bmi,
            use_sex=args.use_sex,
            use_uncertainty_weighting=args.use_uncertainty_weighting,
        )

    return None

class MetricHistoryCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [], "val_kappa": []}

    def on_train_epoch_end(self, trainer, pl_module):
        # Lấy metrics từ trainer.callback_metrics (đã được log)
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            self.history["train_loss"].append(metrics["train_loss"].item())
        if "train_f1" in metrics:
            self.history["train_f1"].append(metrics["train_f1"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "val_loss" in metrics:
            self.history["val_loss"].append(metrics["val_loss"].item())
        if "val_f1" in metrics:
            self.history["val_f1"].append(metrics["val_f1"].item())
        if "val_kappa" in metrics:
            self.history["val_kappa"].append(metrics["val_kappa"].item())

def plot_metrics(history: Dict[str, List[float]], save_dir: str):
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    if history["train_loss"]:
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.plot(epochs, history["train_loss"], label='Train Loss')
    if history["val_loss"]:
        val_epochs = range(1, len(history["val_loss"]) + 1)
        plt.plot(val_epochs, history["val_loss"], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # Plot F1
    plt.figure(figsize=(10, 5))
    if history["train_f1"]:
        epochs = range(1, len(history["train_f1"]) + 1)
        plt.plot(epochs, history["train_f1"], label='Train F1')
    if history["val_f1"]:
        val_epochs = range(1, len(history["val_f1"]) + 1)
        plt.plot(val_epochs, history["val_f1"], label='Val F1')
    plt.title('F1 Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'f1_curve.png'))
    plt.close()
    
    # Plot Kappa
    plt.figure(figsize=(10, 5))
    if history["val_kappa"]:
        val_epochs = range(1, len(history["val_kappa"]) + 1)
        plt.plot(val_epochs, history["val_kappa"], label='Val Kappa')
    plt.title('Kappa over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Kappa')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'kappa_curve.png'))
    plt.close()

# -------------------- TRAIN FUNCTION --------------------
def train(args: argparse.Namespace):

    pl.seed_everything(args.seed, workers=True)

    # ---------- Load base TRAIN ----------
    data_ndarray = load_dataset(args)
    x_full = data_ndarray["x_full"]
    y_full = data_ndarray["y_full"]
    id_groups = data_ndarray["id_groups"]


    args.seq_len = x_full.shape[1]
    print("args.seq_len: ", args.seq_len)

    # ---------- Splitter train- test ----------
    splitter_train_test = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    split_iter_train_test = list(splitter_train_test.split(x_full, y_full, id_groups))

    # ---------- Chọn fold ----------
    if args.fold_index != -1 and not (0 <= args.fold_index < args.n_splits):
        raise ValueError(f"fold_index phải trong [0, {args.n_splits-1}] hoặc = -1")
    selected_folds = range(args.n_splits) if args.fold_index == -1 else [args.fold_index]

    # ---------- Tên run chung & Base directory ----------
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = args.name or f"{args.model}_{run_timestamp}"
    base_exp_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(base_exp_dir, exist_ok=True)

    all_metrics_test: List[Dict[str, float]] = []
    oof_rows: List[Dict[str, Any]] = []   # lưu tất cả VAL-preds của mọi folds

    for fold_idx in selected_folds:
        # --- Index từng fold (outer): train+valid vs test ---
        train_valid_idx, test_idx = split_iter_train_test[fold_idx]

        # Subset tương ứng với phần train+valid
        x_tv = x_full[train_valid_idx]
        y_tv = y_full[train_valid_idx]
        groups_tv = np.array(id_groups)[train_valid_idx]

        # --- Inner split: train / valid trên phần train_valid ---
        splitter_train_valid = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=args.seed
        )  # ~20% valid

        inner_train_idx, inner_valid_idx = next(
            splitter_train_valid.split(x_tv, y_tv, groups_tv)
        )

        # Map về index gốc
        train_idx = train_valid_idx[inner_train_idx]
        valid_idx = train_valid_idx[inner_valid_idx]

        # Lúc này mới index x_full / y_full bằng index gốc
        x_train, y_train = x_full[train_idx], y_full[train_idx]
        y_train_raw = y_train
        x_val, y_val = x_full[valid_idx], y_full[valid_idx]
        x_test, y_test = x_full[test_idx], y_full[test_idx]

        ids_test = np.array(id_groups)[test_idx]
        row_idx_test = test_idx

        # --- Oversampling (nếu bật) ---
        if args.oversampling == "adasyn":
            adasyn = ADASYNWrapper(random_state=args.seed, use_bmi = args.use_bmi, use_sex = args.use_sex)
            x_train, y_train = adasyn.fit_resample(x_train, y_train)
        # elif args.oversampling == "smotenc":
        #     x_train, y_train = smotenc(x_train, y_train)

        # --- DataModule ---
        dm = TSDataModule(
            x_train, y_train, x_val, y_val, x_test, y_test,
            standardize=args.standardize,
            use_weighted_sampler=args.use_weighted_sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        # --- Model ---
        class_weights = dm.class_weights_tensor()
        model = load_model(args, class_weights, y_train_raw)

        # --- Logger ---
        fold_name = f"fold_{fold_idx+1}"
        logger_tb = TensorBoardLogger(save_dir=base_exp_dir, name=fold_name, version="", default_hp_metric=False)
        if args.use_wandb:
            logger_wandb = WandbLogger(project="ECG_Classification_PL", name=f"{exp_name}_{fold_name}")
            logger = [logger_tb, logger_wandb]
        else:
            logger = [logger_tb]

        # --- Callbacks: theo dõi val_kappa ---
        monitor_metric = "val_kappa"
        ckpt = ModelCheckpoint(
            monitor=monitor_metric,
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{" + monitor_metric + ":.4f}",
            auto_insert_metric_name=False,
        )
        early = EarlyStopping(monitor=monitor_metric, mode="max", patience=args.patience)
        lrmon = LearningRateMonitor(logging_interval="epoch")
        history_cb = MetricHistoryCallback()

        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator="auto",
            devices="auto",
            precision=args.precision,
            callbacks=[ckpt, early, lrmon, history_cb],
            logger=logger,
            log_every_n_steps=5,
            deterministic=True,
            inference_mode=False,
        )

        # --- Fit ---
        trainer.fit(model, datamodule=dm, ckpt_path=args.resume_from or None)
        
        # --- Vẽ biểu đồ sau khi train xong ---
        plot_metrics(history_cb.history, logger[0].log_dir)
        print(f"Saved metric plots to: {logger[0].log_dir}")

        # --- Threshold Optimization (for Coral Loss) ---
        optimal_threshold = args.threshold  # Default
        if "coral" in args.model.lower():
            print("Finding optimal threshold on validation set...")
            # Load best model
            best_model = load_model(args, class_weights, y_train_raw)
            checkpoint = torch.load(ckpt.best_model_path, map_location=args.device)
            best_model.load_state_dict(checkpoint['state_dict'])
            best_model.to(args.device)
            best_model.eval()

            # Create Val Dataloader (reusing dm setup)
            val_dls = dm.val_dataloader()
            optimal_threshold = find_optimal_threshold(best_model, val_dls, args.device)
            print(f"Optimal Threshold found: {optimal_threshold:.4f}")

            # Update model threshold
            model.threshold = optimal_threshold
            # Also update best_model if we use it later directly (though pl loads from ckpt usually)
            # We need to make sure 'model' passed to test/predict has the new threshold.
            # Since trainer.test loads from ckpt, we might need to override.
            # Actually, standard PL trainer.test loads from ckpt which HAS OLD THRESHOLD.
            # So we should pass the loaded model with updated threshold.
            
            # Since we loaded best_model above, let's use it.
            best_model.threshold = optimal_threshold
            model = best_model # Switch reference to the loaded best model

        # --- test ở best checkpoint ---
        # Note: ckpt_path=None because we pass the loaded 'model' which has the NEW numeric threshold
        # If we pass ckpt_path, PL reloads weights AND hparams, potentially resetting threshold if it was saved.
        # But 'threshold' is a hparam. So reloading ckpt will reset it to 0.5.
        # We must use the 'model' object directly and NO ckpt_path to use the modification.
        
        test_dls = dm.test_dataloader()
        trainer.test(model=model, dataloaders=test_dls, ckpt_path=None) 


        # --- Lấy metrics (test) ---
        cb = trainer.callback_metrics
        m_test = {
            "fold": fold_idx + 1,
            "test_f1": float(cb.get("test_f1", torch.tensor(float("nan")))),
            "test_loss": float(cb.get("test_loss", torch.tensor(float("nan")))),
            "test_precision": float(cb.get("test_precision", torch.tensor(float("nan")))),
            "test_recall": float(cb.get("test_recall", torch.tensor(float("nan")))),
            "test_accuracy": float(cb.get("test_accuracy", torch.tensor(float("nan")))),
            "test_balanced_acc": float(cb.get("test_balanced_acc", torch.tensor(float("nan")))),
            "test_kappa": float(cb.get("test_kappa", torch.tensor(float("nan")))),
            "log_dir": logger[0].log_dir,
            "best_ckpt": ckpt.best_model_path,
        }
        all_metrics_test.append(m_test)

        # --- Predict OOF (test) bằng best checkpoint ---
        # model.eval() # Trainer handles this
        # Pass model directly to preserve the updated threshold
        preds = trainer.predict(model, dataloaders=test_dls, ckpt_path=None)


        flat = []
        for item in preds:
            flat.extend(item if isinstance(item, list) else [item])

        y_pred = np.concatenate([np.asarray(b["y_pred"]) for b in flat])
        # y_prob = np.concatenate([np.asarray(b["probs"]) for b in flat], axis=0)  # (N, C)
        y_true = np.concatenate([np.asarray(b["y_true"]) for b in flat])

        # an toàn: đảm bảo kích thước khớp với ids_test/row_idx_test
        assert len(y_pred) == len(ids_test) == len(row_idx_test) == len(y_true)

        df_fold = pd.DataFrame({
            "fold": fold_idx + 1,
            "row_index": row_idx_test,
            "ID": ids_test,
            "y_true": y_true,
            "y_pred": y_pred,
        })

        oof_rows.extend(df_fold.to_dict(orient="records"))

        # --- Lưu summary theo fold ---
        summary: Dict[str, Any] = {
            "best_ckpt": ckpt.best_model_path,
            "monitor": monitor_metric,
            "patience": args.patience,
            "log_dir": logger[0].log_dir,
            "in_channels": args.in_channels,
            "nb_classes": args.nb_classes,
            "nb_filters": args.nb_filters,
            "depth": args.depth,
            "kernel_size": args.kernel_size,
            "bottleneck_size": args.bottleneck_size,
            "use_residual": not args.no_residual,
            "use_bottleneck": not args.no_bottleneck,
            "lr": args.lr,
            "average": args.average,
            "standardize": bool(args.standardize),
            "use_weighted_sampler": args.use_weighted_sampler,
            "seed": args.seed,
            "precision": args.precision,
            "fold": fold_idx + 1,
            "optimal_threshold": float(optimal_threshold) if "coral" in args.model.lower() else None,
            "all_metrics_test": m_test,
        }
        with open(os.path.join(logger[0].log_dir, "train_summary.json"), "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)

        print(f"Fold {fold_idx + 1}: logs/checkpoints at: {logger[0].log_dir}")

    # ---------- Tổng hợp kết quả CV ----------
    cv_stats = {}
    if len(all_metrics_test) > 0:
        dfm = pd.DataFrame(all_metrics_test)
        metric_cols = [
            "test_f1", "test_precision", "test_recall", "test_accuracy",
            "test_balanced_acc", "test_kappa", "test_loss",
        ]
        metric_cols = [c for c in metric_cols if c in dfm.columns]
        print("\nCV summary (test):")
        print(dfm[["fold"] + metric_cols].round(4))
        print()
        for c in metric_cols:
            mu = dfm[c].mean()
            sd = dfm[c].std(ddof=1)
            cv_stats[c] = {
                "mean": float(mu),
                "std": float(sd),
                "mean_std": f"{mu:.4f} ± {sd:.4f}"
            }
            print(f"{c}: {mu:.4f} ± {sd:.4f}")

    cv_oof_cm_dir = base_exp_dir
    os.makedirs(cv_oof_cm_dir, exist_ok=True)

    cv_stats_path = os.path.join(cv_oof_cm_dir, f"{exp_name}_cv_stats.json")
    with open(cv_stats_path, "w", encoding="utf-8") as fp:
        json.dump(cv_stats, fp, indent=2)

    # ---------- Gộp & lưu OOF predictions ----------
    df_oof = None
    if len(oof_rows) > 0:
        df_oof = pd.DataFrame(oof_rows)
        # sắp xếp lại đúng thứ tự theo row_index để dễ đối chiếu dữ liệu gốc
        df_oof = df_oof.sort_values(by=["row_index"]).reset_index(drop=True)
        oof_path = os.path.join(cv_oof_cm_dir, f"{exp_name}_oof_predictions.csv")
        df_oof.to_csv(oof_path, index=False, encoding="utf-8")
        print(f"Saved OOF predictions: {oof_path}")

    # ---------- Confusion Matrix CUỐI CÙNG từ OOF ----------
    if df_oof is not None and "y_true" in df_oof.columns and "y_pred" in df_oof.columns:
        y_true_all = df_oof["y_true"].to_numpy()
        y_pred_all = df_oof["y_pred"].to_numpy()

        labels = np.arange(args.nb_classes)
        cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
        cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

        # Vẽ normalized CM
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(labels)
        ax.set_yticks(labels)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix (OOF, normalized)")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = int(cm[i, j])
                percentage = cm_norm[i, j]
                text_label = f"{count}\n({percentage:.1%})"
                color = "white" if percentage > 0.6 else "black"
                ax.text(j, i, text_label, ha="center", va="center", color=color, fontsize=8)
        fig.tight_layout()

        cm_path = os.path.join(cv_oof_cm_dir, f"{exp_name}_confusion_matrix_oof.png")
        fig.savefig(cm_path)
        plt.close(fig)
        print(f"Saved OOF confusion matrix: {cm_path}")

        # --- Log vào W&B nếu có ---
        if args.use_wandb:
            try:
                # 1) log ảnh
                logger_wandb_exps = [lg for lg in trainer.loggers if isinstance(lg, WandbLogger)] if hasattr(trainer, "loggers") else []
                run = logger_wandb_exps[0].experiment if logger_wandb_exps else None
                if run is not None:
                    run.log({"confusion_matrix_oof_image": wandb.Image(cm_path)})
                    # 2) log interactive confusion_matrix
                    class_names = [str(c) for c in labels]
                    cm_plot = wandb.plot.confusion_matrix(
                        y_true=y_true_all.tolist(),
                        preds=y_pred_all.tolist(),
                        class_names=class_names
                    )
                    run.log({"confusion_matrix_oof": cm_plot})
            except Exception as e:
                print(f"[WARN] Không thể log CM lên W&B: {e}")

    return {"fold_metrics": all_metrics_test}



# -------------------- MAIN FUNCTION --------------------
def main():
    parser = argparse.ArgumentParser(description="Train InceptionTime with StratifiedGroupKFold (PyTorch Lightning)")

    # Data sources
    parser.add_argument("--root-dir", type=str, required=True,
                        help="Path to TRAIN/TEST data folder")

    parser.add_argument("--dataset-name", type=str, required=True,
                        help="dataset name")

    parser.add_argument("--model", type=str, required=True,
                        help="model name")

    parser.add_argument( "--use-wandb", action="store_true",
                         help="Sử dụng Weights & Biases để ghi log (Bật nếu cờ này xuất hiện)."
    )

    # Data handling
    parser.add_argument("--no-standardize", action="store_false", dest="standardize", help="Per-channel standardization (TRAIN stats)")
    parser.add_argument("--use-weighted-sampler", default=False, action="store_true", help="Use WeightedRandomSampler on train set")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--oversampling", type=str, default="adasyn")
    parser.add_argument("--use-bmi", action="store_true", help="Include BMI features")
    parser.add_argument("--use-sex", action="store_true", help="Include sex features")

    # CV control
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")  # Sửa: n_splits -> n-splits
    parser.add_argument("--fold-index", type=int, default=-1,
                        help="Fold to train (0..n_splits-1). Use -1 to baseline_model all folds and report mean ± std.")

    # Model
    parser.add_argument("--in-channels", type=int, default=1, help="If None, infer from data")
    parser.add_argument("--nb-classes", type=int, default=4, help="If None, infer from data")
    parser.add_argument("--nb-filters", type=int, default=32)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--kernel-size", type=int, default=41)
    parser.add_argument("--bottleneck-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=1, help="Number of features")
    parser.add_argument("--no-residual", action="store_true")  # Sửa: no_residual -> no-residual
    parser.add_argument("--no-bottleneck", action="store_true")  # Sửa: no_bottleneck -> no-bottleneck
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--average", type=str, default="macro", choices=["macro", "weighted", "micro", "none"])

    # Trainer / logging
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--precision", type=str, default="32", choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--log-dir", dest="log_dir",
                        default="runs")  # Sửa: log_dir -> log-dir (giữ dest="log_dir" để đảm bảo tên thuộc tính)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--resume-from", type=str, default=None)  # Sửa: resume_from -> resume-from
    parser.add_argument("--use-uncertainty-weighting", action="store_true", help="Use uncertainty weighting")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for Coral Loss (default: 0.5)")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
