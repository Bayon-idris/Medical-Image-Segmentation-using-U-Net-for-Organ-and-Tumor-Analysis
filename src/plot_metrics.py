import os
from utils import plot_and_save

def plot_training_curves(train_losses, train_dices, val_dices, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    epochs = list(range(1, len(train_losses) + 1))

    # Loss curve
    plot_and_save(
        x=epochs,
        ys=[train_losses],
        labels=["Train Loss"],
        title="Training Loss",
        xlabel="Epoch",
        ylabel="Loss",
        save_path=f"{save_dir}/loss_curve.png"
    )

    # Dice curve
    plot_and_save(
        x=epochs,
        ys=[train_dices, val_dices],
        labels=["Train Dice", "Val Dice"],
        title="Dice Score",
        xlabel="Epoch",
        ylabel="Dice",
        save_path=f"{save_dir}/dice_curve.png"
    )
