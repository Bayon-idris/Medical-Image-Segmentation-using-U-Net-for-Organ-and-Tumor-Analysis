import numpy as np
import torch
import matplotlib.pyplot as plt


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
        preds = model(images)

    preds = torch.sigmoid(preds)

    # ===== PRENDRE UNE SLICE =====
    img  = images[0, 0].cpu().numpy()
    gt   = masks[0, 0].cpu().numpy()
    pred = (preds[0, 0] > 0.5).cpu().numpy()

    # ===== NORMALISER IMAGE =====
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray")

    # ===== GROUND TRUTH (VERT) =====
    plt.imshow(np.where(gt == 1, 1, np.nan),
               cmap="Greens", alpha=0.4)

    # ===== PREDICTION (ROUGE) =====
    plt.imshow(np.where(pred == 1, 1, np.nan),
               cmap="Reds", alpha=0.6)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Segmentation saved → {save_path}")
