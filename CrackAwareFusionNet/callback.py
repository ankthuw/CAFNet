import os

from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint


class MyPrintingCallBack(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Start Training")

    def on_train_end(self, trainer, pl_module):
        print("Training is done")

    def on_validation_end(self, trainer, pl_module):
        print("Validation completed")


checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", "crack_aware_fusion_net")
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename="cafnet-{epoch:02d}-{val_loss:.4f}",
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
