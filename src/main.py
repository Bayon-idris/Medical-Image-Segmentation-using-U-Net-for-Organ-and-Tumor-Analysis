import json
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from dataset import HeartDataset
from save_prediction import save_segmentation_predictions
from unet import build_unet
from train import train_one_epoch
from evaluate import evaluate
from plot_metrics import plot_training_curves


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_path = Path(
        "D:/Personal Research/computer-vision-project/"
        "Medical-Image-Segmentation-using-U-Net-for-Organ-and-Tumor-Analysis/data/Task02_Heart"
    )

    dataset = HeartDataset(
        images_dir=base_path / "imagesTr", labels_dir=base_path / "labelsTr"
    )

    total = len(dataset)

    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    model = build_unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("🚀 Starting training...")

    epochs = 20
    train_losses = []
    train_dices = []
    val_dices = []

    for epoch in range(epochs):
        loss, dice, _ = train_one_epoch(model, train_loader, optimizer, device)

        val_dice, _ = evaluate(model, val_loader, device)

        train_losses.append(loss)
        train_dices.append(dice)
        val_dices.append(val_dice)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss={loss:.4f} | "
            f"Train Dice={dice:.4f} | "
            f"Val Dice={val_dice:.4f}"
        )

    torch.save(model.state_dict(), "unet_heart.pth")
    print("✅ Model saved")

    print("🧪 Testing model...")
    test_dice, test_iou = evaluate(model, test_loader, device)
    print(f"Test Dice={test_dice:.4f} | Test IoU={test_iou:.4f}")

    plot_training_curves(train_losses, train_dices, val_dices, save_dir="outputs")
    save_segmentation_predictions(
        model, test_loader, device, save_path="outputs/segmentation_overlay.png"
    )

    metrics = {
        "train_loss": train_losses,
        "train_dice": train_dices,
        "val_dice": val_dices,
    }

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
