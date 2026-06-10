import os

import torch

import config
from dataloader import get_loaders
from model import CrackAwareFusionNet
from utils import eval_metrics, save_predictions_as_imgs


def main():
    train_loader, val_loader, test_loader = get_loaders(
        config.TRAIN_IMG_DIR,
        config.TRAIN_MASK_DIR,
        config.VAL_IMG_DIR,
        config.VAL_MASK_DIR,
        config.TEST_IMG_DIR,
        config.TEST_MASK_DIR,
        1,
        config.NUM_WORKERS,
        config.PIN_MEMORY,
    )

    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    model = CrackAwareFusionNet(
        attn_gate=config.ATTN_GATE,
        crackam=config.CRACKAM,
        crackspam=config.CRACKSPAM,
    ).to(config.DEVICE)

    print("Loading Model")

    ck_file_path = config.CHECKPOINTS_PATH
    state_dict = torch.load(ck_file_path, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    mul_outputs = True
    mode = "test"

    print()
    print("Computing Metrics")
    eval_metrics(loader=dataloaders[mode], model=model, multiple_outputs=mul_outputs)
    print("-----------------------------")

    print("Saving Images")
    file_name = "RECALL_outputs"
    current_path = os.getcwd()
    if file_name not in os.listdir(current_path):
        os.makedirs(file_name)
    save_predictions_as_imgs(
        dataloaders[mode],
        model,
        folder=file_name + "/",
        device=config.DEVICE,
        multiple_outputs=mul_outputs,
    )
    print("Saved all images")


if __name__ == "__main__":
    main()
