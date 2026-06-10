import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

import config
from callback import MyPrintingCallBack, checkpoint_callback, early_stopping
from dataloader import get_loaders
from model import CrackAwareFusionNet

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders(
        config.TRAIN_IMG_DIR,
        config.TRAIN_MASK_DIR,
        config.VAL_IMG_DIR,
        config.VAL_MASK_DIR,
        config.TEST_IMG_DIR,
        config.TEST_MASK_DIR,
        config.BATCH_SIZE,
        config.NUM_WORKERS,
        config.PIN_MEMORY,
    )

    logger = TensorBoardLogger(
        save_dir="logs",
        name="cafnet",
        version="baseline",
    )

    model = CrackAwareFusionNet(
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        attn_gate=config.ATTN_GATE,
        crackam=config.CRACKAM,
        crackspam=config.CRACKSPAM,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision="16-mixed",
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

    trainer.fit(model, train_loader, val_loader)

    if trainer.global_rank == 0:
        print("\nBest checkpoint:")
        print(checkpoint_callback.best_model_path)

    trainer.validate(
        model,
        val_loader,
        ckpt_path=checkpoint_callback.best_model_path,
    )

    trainer.test(
        model,
        test_loader,
        ckpt_path=checkpoint_callback.best_model_path,
    )

    if trainer.global_rank == 0:
        outputs_dir = os.path.join(os.getcwd(), "outputs")
        ckpt_dir = os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        state_path = os.path.join(outputs_dir, "final_cafnet.pth")
        torch.save(model.state_dict(), state_path)
        print(f"Saved final model state_dict to {state_path}")
