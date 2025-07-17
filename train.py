import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CNNBiLSTMSecondaryStructure
from dataset import ProteinDataset
from config import TRAIN_CSV, VALID_CSV, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE
import os
import numpy as np

def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)                      # [B, L, 3]
        out = out.permute(0, 2, 1)         # [B, 3, L]
        loss = criterion(out, y)           # y: [B, L]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)                        # [B, L, 3]
            loss = criterion(out.permute(0, 2, 1), y)

            preds = out.argmax(dim=2)             # [B, L]
            mask = (y != -100)                    # Ignore padding
            correct += ((preds == y) & mask).sum().item()
            total += mask.sum().item()
            total_loss += loss.item()
    acc = correct / total if total > 0 else 0.0
    return total_loss / len(loader), acc

def main():
    print("Device:", DEVICE)

    model = CNNBiLSTMSecondaryStructure().to(DEVICE)
    train_set = ProteinDataset(TRAIN_CSV)
    val_set = ProteinDataset(VALID_CSV)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    # Class weights based on confusion matrix distribution
    class_weights = torch.tensor([0.33, 0.59, 0.075], dtype=torch.float32).to(DEVICE)
    print(f"Using class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    patience = 7
    wait = 0

    os.makedirs("outputs", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step(val_loss)

        print(f"[{epoch}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies
            }, "model_final.pt")
            print(f"✅ Saved new best model at epoch {epoch}")
        else:
            wait += 1
            print(f" No improvement ({wait}/{patience})")

        if wait >= patience:
            print("Early stopping triggered.")
            break

    # Save .npy files
    np.save("outputs/train_losses.npy", np.array(train_losses))
    np.save("outputs/val_losses.npy", np.array(val_losses))
    np.save("outputs/val_accuracies.npy", np.array(val_accuracies))


if __name__ == "__main__":
    main()