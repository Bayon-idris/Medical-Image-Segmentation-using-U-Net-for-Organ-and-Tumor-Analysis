import torch

def dice_score(pred, target, smooth=1e-6):
    """
    Dice Score pour multi-classes.
    pred: [B,C,H,W]
    target: [B,H,W] (indices de classes)
    """
    pred_classes = torch.argmax(pred, dim=1) 

    intersection = (pred_classes == target).sum().float()
    return (2 * intersection + smooth) / (pred_classes.numel() + smooth)



def iou_score(pred, target, smooth=1e-6):
    pred_classes = torch.argmax(pred, dim=1)
    intersection = (pred_classes == target).sum().float()
    union = pred_classes.numel() + target.numel() - intersection
    return (intersection + smooth) / (union + smooth)

