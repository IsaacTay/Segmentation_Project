import matplotlib.pyplot as plt
from pathlib import Path

def plot_losses(train_losses, val_losses, root="plots/"):
    root = Path(root)
    plt.plot([loss for loss in train_losses], label='Training Loss')
    plt.plot([loss for loss in val_losses], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses Over Epochs')
    plt.legend()
    plt.savefig(root / 'losses.png')

def plot_val_metrics(f1s, ious, dice_coefficients, root="plots/"):
    root = Path(root)
    plt.plot(f1s, label='Validation Pixel F1')
    plt.plot(ious, label='Validation IOU')
    plt.plot(dice_coefficients, label='Validation DICE')

    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Validation Metrics Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(root / 'val_metrics.png')

def plot_train_metrics(tf1, tiou, tdice, root="plots/"):
    root = Path(root)
    plt.plot(tf1, label='Training Pixel F1s')
    plt.plot(tiou, label='Training IOU')
    plt.plot(tdice, label='Training DICE')

    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Training Metrics Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(root / 'train_metrics.png')

def plot_f1(f1s, tf1, root="plots/"):
    root = Path(root)
    plt.plot(f1s, label='Validation Pixel F1')
    plt.plot(tf1, label='Training Pixel F1')

    plt.xlabel('Epoch')
    plt.ylabel('F1 Value')
    plt.title('F1s Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(root / 'f1.png')

def plot_iou(ious, tiou, root="plots/"):
    root = Path(root)
    plt.plot(ious, label='Validation IOU')
    plt.plot(tiou, label='Training IOU')

    plt.xlabel('Epoch')
    plt.ylabel('IOU Value')
    plt.title('IOU Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(root / 'iou.png')

def plot_dice(dice_coefficients, tdice, root="plots/"):
    root = Path(root)
    plt.plot(dice_coefficients, label='Validation DICE')
    plt.plot(tdice, label='Training DICE')

    plt.xlabel('Epoch')
    plt.ylabel('DICE Value')
    plt.title('DICE Coefficients Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(root / 'dice.png')
