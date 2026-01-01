from pytorch_lightning.callbacks import EarlyStopping, Callback, ModelCheckpoint
import os

class MyPrintingCallBack(Callback):
    def __init__(self):
        super(MyPrintingCallBack, self).__init__()

    def on_train_start(self, trainer, pl_module):
        print("Start Training")

    def on_train_end(self, trainer, pl_module):
        print("Training is done")
        
    def on_validation_end(self, trainer, pl_module):
        print("Validation completed")

# create a directory for saving checkpoints
checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints', 'hybrid_model_bifusion')
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename='hybrid-{epoch:02d}-{val_loss:.4f}',
    verbose=True,
    save_last=True,
    save_top_k=3,  # Save the top 3 models based on validation loss
    monitor='val_loss',
    mode='min'
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5, 
    verbose=True,
    mode='min',
    min_delta=1e-4  
)
