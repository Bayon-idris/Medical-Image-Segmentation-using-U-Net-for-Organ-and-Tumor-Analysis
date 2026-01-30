import torch
from metrics import dice_score, iou_score

def evaluate(model, loader, device):
    model.eval()

    dice, iou = 0, 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            preds = model(images)

            dice += dice_score(preds, masks).item()
            iou  += iou_score(preds, masks).item()

    n = len(loader)
    return dice/n, iou/n
