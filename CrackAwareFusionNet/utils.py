import os

import matplotlib.pyplot as plt
import torch
import torchvision

result_save_ind = 0
threshold = 0.5


def eval_metrics(loader, model, device="cuda", multiple_outputs=False):
    model.eval()
    eps = 1e-7

    TP_total = 0
    FP_total = 0
    TN_total = 0
    FN_total = 0

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            if multiple_outputs:
                final_output = model(x)[result_save_ind]
                preds_probability = torch.sigmoid(final_output)
                preds = (preds_probability > threshold).float()
            else:
                preds_probability = torch.sigmoid(model(x))
                preds = (preds_probability > threshold).float()

            confusion_matirx = preds / y

            TP = torch.sum(confusion_matirx == 1).item()
            FP = torch.sum(confusion_matirx == float("inf")).item()
            TN = torch.sum(torch.isnan(confusion_matirx)).item()
            FN = torch.sum(confusion_matirx == 0).item()

            TP_total += TP
            FP_total += FP
            TN_total += TN
            FN_total += FN

    accuracy = (TP_total + TN_total) / (TP_total + FP_total + TN_total + FN_total + eps)
    precision = TP_total / (TP_total + FP_total + eps)
    recall = TP_total / (TP_total + FN_total + eps)
    FP_rate = FP_total / (FP_total + TN_total + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)
    dice_score = 2 * TP_total / (2 * TP_total + FP_total + FN_total + eps)
    IOU_score = TP_total / (TP_total + FP_total + FN_total + eps)

    print(
        f"Global Accuracy : {accuracy} / Precision : {precision} / "
        f"Recall : {recall} / FPR : {FP_rate} / F1 score : {f1_score}"
    )
    print(f"Dice Score {dice_score} / IOU score {IOU_score}")


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda", multiple_outputs=False
):
    model.eval()

    for x, y, filenames in loader:
        x = x.to(device=device)

        with torch.no_grad():
            if multiple_outputs:
                final_output = model(x)[result_save_ind]
                preds_probability = torch.sigmoid(final_output)
                preds = (preds_probability > threshold).float()
            else:
                preds_probability = torch.sigmoid(model(x))
                preds = (preds_probability > threshold).float()

        for i, filename in enumerate(filenames):
            base_name = os.path.splitext(filename)[0]
            pred_filename = f"{folder}/pred_{base_name}.png"
            gt_filename = f"{folder}/gt_{base_name}.png"

            torchvision.utils.save_image(preds[i].unsqueeze(0), pred_filename)
            torchvision.utils.save_image(y[i].unsqueeze(0).unsqueeze(0), gt_filename)

    model.train()
    print(f"Saved all image with original names to {folder}")


def loss_plot(train_loss, val_loss):
    if len(train_loss) != len(val_loss):
        print("The number of losses are different")
    else:
        labels = [i for i in range(1, len(train_loss) + 1)]
        plt.plot(train_loss)
        plt.plot(val_loss)
        ticks = [i for i in range(0, len(train_loss), 10)]
        tick_labels = [labels[i - 1] for i in ticks]
        plt.xticks(ticks, tick_labels)
        plt.gca().get_xticklabels()[0].set_visible(False)
        plt.xlabel("Epoch", fontsize=17)
        plt.ylabel("Loss", fontsize=17)
        plt.show()
        plt.savefig("loss_output.png")
