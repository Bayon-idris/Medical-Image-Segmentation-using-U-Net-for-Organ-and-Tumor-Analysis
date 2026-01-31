import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def save_segmentation_predictions(
    model,
    loader,
    device,
    save_path="outputs/segmentation_result.png"
):
    model.eval()

    images, masks = next(iter(loader))
    images = images.to(device)
    masks  = masks.to(device)

    with torch.no_grad():
        preds = model(images)           # [B, C, H, W]
        preds = torch.argmax(preds, 1)  # [B, H, W]

    # ===== PRENDRE UNE IMAGE =====
    img  = images[0, 0].cpu().numpy()
    gt   = masks[0].cpu().numpy()
    pred = preds[0].cpu().numpy()

    # ===== NORMALISER IMAGE =====
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # ===== COLORMAP DISCRÈTE =====
    # 0 = background, 1 = organe, 2 = tumeur
    cmap = ListedColormap([
        (0, 0, 0, 0.0),  # background → transparent
        (0, 0, 1, 0.85), # organe → bleu
        (1, 0, 0, 0.85), # tumeur → rouge
    ])

    # ===== AFFICHAGE =====
    plt.figure(figsize=(12, 6))

    # ---- Ground Truth ----
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.imshow(img, cmap="gray")
    plt.imshow(gt, cmap=cmap, interpolation="nearest")
    plt.axis("off")

    # ---- Prediction ----
    plt.subplot(1, 2, 2)
    plt.title("Prediction")
    plt.imshow(img, cmap="gray")
    plt.imshow(pred, cmap=cmap, interpolation="nearest")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Segmentation saved → {save_path}")
