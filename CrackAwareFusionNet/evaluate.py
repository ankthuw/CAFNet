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
    def __init__(self, output_dir=None, save_mask=False, threshold=0.5, eps=1e-6):
        super().__init__()
        self.save_mask = save_mask
        self.threshold = threshold
        self.eps = eps
        self.output_dir = Path(output_dir) if output_dir else None

        if self.save_mask and self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.reset()

    def reset(self):
        self.total_iou = 0.0
        self.total_dice = 0.0
        self.total_precision = 0.0
        self.total_recall = 0.0
        self.total_specificity = 0.0
        self.num_samples = 0

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).bool()
        gt_masks = batch[1].bool().unsqueeze(1)

        preds_flat = preds.view(preds.size(0), -1)
        gt_flat = gt_masks.view(gt_masks.size(0), -1)

        tp = (preds_flat & gt_flat).float().sum(dim=1)
        fp = (preds_flat & ~gt_flat).float().sum(dim=1)
        fn = (~preds_flat & gt_flat).float().sum(dim=1)
        tn = (~preds_flat & ~gt_flat).float().sum(dim=1)

        iou_scores = (tp + self.eps) / (tp + fp + fn + self.eps)
        dice_scores = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        precision_scores = (tp + self.eps) / (tp + fp + self.eps)
        recall_scores = (tp + self.eps) / (tp + fn + self.eps)
        specificity_scores = (tn + self.eps) / (tn + fp + self.eps)

        self.total_iou += iou_scores.sum().item()
        self.total_dice += dice_scores.sum().item()
        self.total_precision += precision_scores.sum().item()
        self.total_recall += recall_scores.sum().item()
        self.total_specificity += specificity_scores.sum().item()
        self.num_samples += iou_scores.size(0)

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
    def mean_iou(self):
        return self.total_iou / self.num_samples if self.num_samples else 0.0

    @property
    def mean_dice(self):
        return self.total_dice / self.num_samples if self.num_samples else 0.0

    @property
    def mean_precision(self):
        return self.total_precision / self.num_samples if self.num_samples else 0.0

    @property
    def mean_recall(self):
        return self.total_recall / self.num_samples if self.num_samples else 0.0

    @property
    def mean_specificity(self):
        return self.total_specificity / self.num_samples if self.num_samples else 0.0


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


def load_model(ckpt_path, map_location="cpu"):
    ckpt_path = str(ckpt_path)
    if ckpt_path.endswith(".ckpt"):
        return EvalCrackAwareFusionNet.load_from_checkpoint(
            ckpt_path,
            map_location=map_location,
            weights_only=False,
        )
    if ckpt_path.endswith(".pth"):
        model = EvalCrackAwareFusionNet()
        state_dict = torch.load(ckpt_path, map_location=map_location, weights_only=True)
        model.load_state_dict(state_dict)
        return model
    raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")


def print_metrics(callback):
    print("\n" + "=" * 60)
    print("COMPREHENSIVE METRICS")
    print("=" * 60)
    print(f"  Samples evaluated : {callback.num_samples}")
    print("-" * 60)
    print(f"  mIoU (Jaccard)    : {callback.mean_iou:.4f}")
    print(f"  Dice (F1-Score)   : {callback.mean_dice:.4f}")
    print(f"  Precision         : {callback.mean_precision:.4f}")
    print(f"  Recall            : {callback.mean_recall:.4f}")
    print(f"  Specificity       : {callback.mean_specificity:.4f}")
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
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.ckpt)
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
