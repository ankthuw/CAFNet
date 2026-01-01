from dataloader import get_loaders
import pytorch_lightning as pl
import config
from model import CrackAwareFusionNet
import torch
from callback import MyPrintingCallBack, checkpoint_callback, early_stopping
import os
from pytorch_lightning.loggers import CSVLogger



torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders(
        config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR,
        config.VAL_IMG_DIR, config.VAL_MASK_DIR,
        config.TEST_IMG_DIR, config.TEST_MASK_DIR,
        config.BATCH_SIZE, config.NUM_WORKERS, config.PIN_MEMORY,
    )   
    logger_name = "crack_aware_fusion_net"
    logger = CSVLogger(save_dir="logs", name=logger_name)

    model = CrackAwareFusionNet(learning_rate=config.LEARNING_RATE).to(config.DEVICE)

    trainer = pl.Trainer(
        logger=logger,
        enable_checkpointing=True,
        accelerator="gpu",
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision="16-mixed",
        callbacks=[MyPrintingCallBack(), checkpoint_callback, early_stopping],
        enable_model_summary=False,
        log_every_n_steps=10,
    )

    # Training
    trainer.fit(model, train_loader, val_loader)

    # Validation
    trainer.validate(model, val_loader)

    # Testing with best checkpoint
    trainer.test(model, test_loader, ckpt_path="best")

    # Save final model (fixed filenames, no timestamp)
    outputs_dir = os.path.join(os.getcwd(), "outputs")
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    state_path = os.path.join(outputs_dir, f"final_{logger_name}.pth")
    torch.save(model.state_dict(), state_path)
    print(f"Saved final model state_dict to {state_path}")

    ckpt_path = os.path.join(ckpt_dir, f"{logger_name}.ckpt")
    trainer.save_checkpoint(ckpt_path)
    print(f"Saved trainer checkpoint to {ckpt_path}")
