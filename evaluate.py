import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from dataset import ProteinDataset
from model import CNNBiLSTMSecondaryStructure
from config import (
    VALID_CSV, TEST_CB513, TEST_TS115, TEST_CASP12,
    BATCH_SIZE, DEVICE
)

def evaluate_dataset(name, path, model, save_conf_matrix=False):
    dataset = ProteinDataset(path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            preds = out.argmax(dim=2)
            mask = (y != -100)

            all_preds.extend(preds[mask].cpu().numpy())
            all_labels.extend(y[mask].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean() * 100
    report = classification_report(all_labels, all_preds, target_names=["H", "E", "C"])

    # Save classification report
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name.lower()}_report.txt", "w") as f:
        f.write(f"{name} Accuracy: {acc:.2f}%\n\n")
        f.write(report)

    print(f"\n✔ {name} Evaluation Done - Accuracy: {acc:.2f}%")
    print(report)

    # Save confusion matrix
    if save_conf_matrix:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["H", "E", "C"], yticklabels=["H", "E", "C"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(f"results/{name.lower()}_confusion_matrix.png")
        plt.close()

    # Save raw predictions if needed
    if name.lower() == "validation":
        np.save("results/val_predictions.npy", all_preds)
        np.save("results/val_labels.npy", all_labels)

def plot_loss_curve(checkpoint_path, save_path="results/loss_curve.png"):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    val_accuracies = [acc * 100 for acc in checkpoint['val_accuracies']]
    epochs = list(range(1, len(train_losses) + 1))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, train_losses, label="Train Loss", color="tab:blue", marker="o")
    ax1.plot(epochs, val_losses, label="Val Loss", color="tab:orange", marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_accuracies, label="Val Accuracy (%)", color="tab:green", marker="s")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.legend(loc="upper right")

    plt.title("Training & Validation Loss + Accuracy")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"📉 Saved loss/accuracy plot to {save_path}")

if __name__ == "__main__":
    model = CNNBiLSTMSecondaryStructure().to(DEVICE)
    checkpoint_path = "model_final.pt"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Plot loss & accuracy
    plot_loss_curve(checkpoint_path)

    # Evaluate datasets
    evaluate_dataset("Validation", VALID_CSV, model, save_conf_matrix=True)
    evaluate_dataset("CB513", TEST_CB513, model)
    evaluate_dataset("TS115", TEST_TS115, model)
    evaluate_dataset("CASP12", TEST_CASP12, model)
