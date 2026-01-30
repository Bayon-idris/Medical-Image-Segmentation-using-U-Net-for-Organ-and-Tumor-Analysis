import torch
import torch.nn as nn
from metrics import dice_score, iou_score

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss, total_dice, total_iou = 0, 0, 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        preds = model(images)

        loss = loss_fn(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(preds, masks).item()
        total_iou += iou_score(preds, masks).item()

    n = len(loader)
    return total_loss/n, total_dice/n, total_iou/n
