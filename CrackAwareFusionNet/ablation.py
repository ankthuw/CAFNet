import argparse
import csv
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

import config
from callback import MyPrintingCallBack
from dataloader import get_loaders
from model import CrackAwareFusionNet

FUSION_VARIANTS = ["add", "concat", "attention", "bilinear", "cabm"]


def resolve_dataset_paths(dataset_root):
    root = Path(dataset_root)
    return {
        "train_img": str(root / "train" / "IMG"),
        "train_mask": str(root / "train" / "GT"),
        "val_img": str(root / "val" / "IMG"),
        "val_mask": str(root / "val" / "GT"),
        "test_img": str(root / "test" / "IMG"),
        "test_mask": str(root / "test" / "GT"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run CAFNet fusion ablations sequentially")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=FUSION_VARIANTS,
        choices=FUSION_VARIANTS,
        help="Fusion variants to run in order",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=config.dataset or None,
        help="Root folder containing train/val/test subfolders. Use /kaggle/input/<dataset-name> on Kaggle.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path.cwd() / "outputs" / "ablation"),
        help="Directory for ablation outputs and saved checkpoints",
    )
    return parser.parse_args()


def build_trainer(variant, epochs, output_dir):
    checkpoint_dir = Path(output_dir) / "checkpoints" / variant
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = TensorBoardLogger(
        save_dir=str(Path(output_dir) / "logs"),
        name="cafnet_ablation",
        version=variant,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"{variant}-{{epoch:02d}}-{{val_loss:.4f}}",
        verbose=True,
        save_last=True,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=True,
        mode="min",
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = "16-mixed" if accelerator == "gpu" else 32

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        min_epochs=1,
        max_epochs=epochs,
        precision=precision,
        log_every_n_steps=20,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[
            MyPrintingCallBack(),
            checkpoint_callback,
            early_stopping,
            LearningRateMonitor(logging_interval="epoch"),
            TQDMProgressBar(refresh_rate=10),
        ],
        enable_model_summary=False,
        enable_progress_bar=True,
    )

    return trainer, checkpoint_callback


def run_variant(variant, epochs, batch_size, num_workers, dataset_root, output_dir):
    dataset_paths = resolve_dataset_paths(dataset_root)

    train_loader, val_loader, test_loader = get_loaders(
        dataset_paths["train_img"],
        dataset_paths["train_mask"],
        dataset_paths["val_img"],
        dataset_paths["val_mask"],
        dataset_paths["test_img"],
        dataset_paths["test_mask"],
        batch_size,
        num_workers,
        config.PIN_MEMORY,
    )

    model = CrackAwareFusionNet(
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        attn_gate=config.ATTN_GATE,
        crackam=config.CRACKAM,
        crackspam=config.CRACKSPAM,
        fusion_type=variant,
    )

    trainer, checkpoint_callback = build_trainer(variant, epochs, output_dir)

    print("\n" + "=" * 80)
    print(f"Running fusion ablation: {variant}")
    print("=" * 80)

    trainer.fit(model, train_loader, val_loader)

    best_path = checkpoint_callback.best_model_path
    print(f"Best checkpoint for {variant}: {best_path}")

    val_results = trainer.validate(
        model,
        val_loader,
        ckpt_path=best_path if best_path else None,
    )
    test_results = trainer.test(
        model,
        test_loader,
        ckpt_path=best_path if best_path else None,
    )

    outputs_dir = Path(output_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if best_path:
        best_model = CrackAwareFusionNet.load_from_checkpoint(best_path, fusion_type=variant)
        state_path = outputs_dir / f"{variant}.pth"
        torch.save(best_model.state_dict(), state_path)
        print(f"Saved best state_dict to {state_path}")

    metrics_row = {"fusion_type": variant, "best_checkpoint": best_path}
    if val_results:
        metrics_row.update({f"val_{k}": v for k, v in val_results[0].items()})
    if test_results:
        metrics_row.update({f"test_{k}": v for k, v in test_results[0].items()})

    return metrics_row


def write_results_csv(rows, output_path):
    if not rows:
        return

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not args.dataset_root:
        raise ValueError(
            "dataset_root is required. On Kaggle this should point to the mounted dataset, for example /kaggle/input/<dataset-name>."
        )
    pl.seed_everything(args.seed, workers=True)

    results = []
    for variant in args.variants:
        result = run_variant(
            variant=variant,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
        )
        results.append(result)

    summary_path = Path(args.output_dir) / "ablation_results.csv"
    write_results_csv(results, summary_path)
    print(f"\nSaved ablation summary to {summary_path}")


if __name__ == "__main__":
    main()