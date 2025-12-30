import torch
import os
from model import CrackAwareFusionNet
from utils import save_predictions_as_imgs, eval_metrics
from dataloader import get_loaders
import config


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
    model = CrackAwareFusionNet().to(config.DEVICE)

    print('Loading Model')

    ck_file_path = config.CHECKPOINTS_PATH
    state_dict = torch.load(ck_file_path, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    mul_outputs = True
    mode = 'test'

    print()
    print('Computing Metrics')
    eval_metrics(loader=dataloaders[mode], model=model, multiple_outputs=mul_outputs)
    print('-----------------------------')


    print('Saving Images')
    file_name = 'RECALL_outputs'
    current_path = os.getcwd()
    if file_name not in os.listdir(os.path.join(current_path)):
        os.makedirs(file_name)
    save_predictions_as_imgs(dataloaders[mode], model, folder=file_name+"/", device=config.DEVICE, multiple_outputs=mul_outputs)
    print('Saved all images')


if __name__ == "__main__":
    main()
