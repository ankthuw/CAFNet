import argparse
import os
import shutil
from pathlib import Path

import albumentations as A
import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from dataset import CrackDataset
from model import CrackAwareFusionNet

IMAGE_SIZE = (256, 256)
MEAN = [0.51789941, 0.51360926, 0.547762]
STD = [0.1812099, 0.17746663, 0.20386334]


class ComprehensiveMetricsAndSaveCallback(pl.Callback):
    """Global pixel-wise metrics (same logic as utils.eval_metrics)."""

    def __init__(self, output_dir=None, save_mask=False, threshold=0.5, eps=1e-7):
        super().__init__()
        self.save_mask = save_mask
        self.threshold = threshold
        self.eps = eps
        self.output_dir = Path(output_dir) if output_dir else None

        if self.save_mask and self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.reset()

    def reset(self):
        self.tp_total = 0.0
        self.fp_total = 0.0
        self.fn_total = 0.0
        self.tn_total = 0.0
        self.num_samples = 0

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).bool()
        gt_masks = batch[1].bool().unsqueeze(1)

        preds_flat = preds.view(preds.size(0), -1)
        gt_flat = gt_masks.view(gt_masks.size(0), -1)

        self.tp_total += (preds_flat & gt_flat).float().sum().item()
        self.fp_total += (preds_flat & ~gt_flat).float().sum().item()
        self.fn_total += (~preds_flat & gt_flat).float().sum().item()
        self.tn_total += (~preds_flat & ~gt_flat).float().sum().item()
        self.num_samples += preds.size(0)

        if not self.save_mask or self.output_dir is None:
            return

        masks_to_save = preds.float()
        filenames = batch[2] if len(batch) >= 3 else [
            f"batch{batch_idx}_img{i}" for i in range(masks_to_save.size(0))
        ]

        for i, mask in enumerate(masks_to_save):
            name = Path(filenames[i]).stem
            save_path = self.output_dir / f"{name}_pred.png"
            vutils.save_image(mask, save_path)

    @property
    def precision(self):
        return self.tp_total / (self.tp_total + self.fp_total + self.eps)

    @property
    def recall(self):
        return self.tp_total / (self.tp_total + self.fn_total + self.eps)

    @property
    def specificity(self):
        return self.tn_total / (self.tn_total + self.fp_total + self.eps)

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r + self.eps)

    @property
    def dice(self):
        return 2 * self.tp_total / (2 * self.tp_total + self.fp_total + self.fn_total + self.eps)

    @property
    def iou(self):
        return self.tp_total / (self.tp_total + self.fp_total + self.fn_total + self.eps)

    # Backward-compatible aliases used by notebooks
    @property
    def mean_precision(self):
        return self.precision

    @property
    def mean_recall(self):
        return self.recall

    @property
    def mean_specificity(self):
        return self.specificity

    @property
    def mean_dice(self):
        return self.dice

    @property
    def mean_iou(self):
        return self.iou


class EvalCrackAwareFusionNet(CrackAwareFusionNet):
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, _, _ = batch
        return self.forward(imgs)[0]


class SingleFolderDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, img_size=IMAGE_SIZE, batch_size=16, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.transform = A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])

    def setup(self, stage=None):
        img_dir = os.path.join(self.data_dir, "IMG")
        gt_dir = os.path.join(self.data_dir, "GT")
        if not os.path.isdir(img_dir) or not os.path.isdir(gt_dir):
            raise FileNotFoundError(f"Missing IMG/ or GT/ in {self.data_dir}")
        self.test_dataset = CrackDataset(img_dir, gt_dir, self.transform)

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def load_model(ckpt_path, map_location="cpu", fusion_type="cabm"):
    ckpt_path = str(ckpt_path)
    if ckpt_path.endswith(".ckpt"):
        return EvalCrackAwareFusionNet.load_from_checkpoint(
            ckpt_path,
            map_location=map_location,
            fusion_type=fusion_type,
            weights_only=False,
        )
    if ckpt_path.endswith(".pth"):
        model = EvalCrackAwareFusionNet(fusion_type=fusion_type)
        state_dict = torch.load(ckpt_path, map_location=map_location, weights_only=True)
        model.load_state_dict(state_dict)
        return model
    raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")


def print_metrics(callback):
    print("\n" + "=" * 60)
    print("GLOBAL METRICS (utils.eval_metrics style)")
    print("=" * 60)
    print(f"  Images evaluated  : {callback.num_samples}")
    print("-" * 60)
    print(f"  mIoU (Jaccard)    : {callback.iou:.4f}")
    print(f"  Dice (F1-Score)   : {callback.dice:.4f}")
    print(f"  Precision         : {callback.precision:.4f}")
    print(f"  Recall            : {callback.recall:.4f}")
    print(f"  Specificity       : {callback.specificity:.4f}")
    print(f"  F1 check 2PR/(P+R): {callback.f1:.4f}")
    print("=" * 60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CAFNet on a single IMG/GT folder")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt or .pth")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder containing IMG/ and GT/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_mask", action="store_true")
    parser.add_argument("--mask_dir", type=str, default="predicted_masks")
    parser.add_argument("--zip_masks", action="store_true")
    parser.add_argument("--fusion_type", type=str, default="cabm", choices=["add", "concat", "attention", "bilinear", "cabm"])
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.ckpt, fusion_type=args.fusion_type)
    datamodule = SingleFolderDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    eval_callback = ComprehensiveMetricsAndSaveCallback(
        output_dir=args.mask_dir,
        save_mask=args.save_mask,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        callbacks=[eval_callback],
        enable_progress_bar=True,
    )

    print(f"\nEvaluating: {args.data_dir}")
    trainer.predict(model=model, datamodule=datamodule, ckpt_path=None)
    print_metrics(eval_callback)

    if args.save_mask and args.zip_masks:
        print("Zipping predicted masks...")
        shutil.make_archive(args.mask_dir, "zip", args.mask_dir)
        shutil.rmtree(args.mask_dir)
        print(f"Saved: {args.mask_dir}.zip")


if __name__ == "__main__":
    main()
