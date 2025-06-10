import os
import json
import glob
import argparse
from datetime import datetime

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

from pymino import clear_full_rows, clear_full_cols, clear_full_blocks

# Utility functions and dataset (same as before)

class PyMinoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 9 * 9, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 243)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def train_one_epoch(model, dataloader, optimizer, device, scaler, warmup_scheduler=None):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction='none')
    for x_batch, y_batch, weights in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        weights = weights.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(x_batch)
            losses = criterion(logits, y_batch)
            weighted_loss = (losses * weights).mean()
        scaler.scale(weighted_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if warmup_scheduler:
            warmup_scheduler.step()
        total_loss += weighted_loss.item() * x_batch.size(0)
    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for x_batch, y_batch, _ in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_folder', type=str, default='logs')
    parser.add_argument('--epochs1', type=int, default=80)
    parser.add_argument('--lr1', type=float, default=2e-4)
    parser.add_argument('--warmup_epochs1', type=int, default=5)
    parser.add_argument('--epochs2', type=int, default=80)
    parser.add_argument('--lr2', type=float, default=3e-4)
    parser.add_argument('--warmup_epochs2', type=int, default=5)
    parser.add_argument('--epochs3', type=int, default=80)
    parser.add_argument('--lr3', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs3', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--minimum_score', type=int, default=500)
    parser.add_argument('--max_score_cutoff', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = PyMinoNet().to(device)
    scaler = torch.amp.GradScaler()

    # Load datasets
    from pymino_dataset import load_train_val_datasets
    dl1_train, dl1_val, dl2_train, dl2_val, dl3_train, dl3_val = load_train_val_datasets(
        logs_folder=args.logs_folder,
        batch_size=args.batch_size,
        minimum_score=args.minimum_score,
        max_score_cutoff=args.max_score_cutoff
    )

    for step, (epochs, lr, warmup_epochs, train_dl, val_dl) in enumerate([
        (args.epochs1, args.lr1, args.warmup_epochs1, dl1_train, dl1_val),
        (args.epochs2, args.lr2, args.warmup_epochs2, dl2_train, dl2_val),
        (args.epochs3, args.lr3, args.warmup_epochs3, dl3_train, dl3_val)
    ], start=1):

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        total_steps = len(train_dl) * warmup_epochs
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min(1.0, step / total_steps) if total_steps > 0 else 1.0)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True)

        best_acc = 0.0
        no_improve = 0

        print(f"\n=== Training Step {step} (LR={lr}) ===")

        for epoch in range(1, epochs + 1):
            loss = train_one_epoch(model, train_dl, optimizer, device, scaler, warmup_scheduler)
            acc = validate(model, val_dl, device)
            print(f"Epoch {epoch}/{epochs}  Loss = {loss:.4f}  ValAcc = {acc*100:.2f}%")
            scheduler.step(acc)

            if acc > best_acc:
                best_acc = acc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 7:
                    print(f"No improvement in Step {step} for 7 epochs, stopping early.")
                    break

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    torch.save(model.state_dict(), f"pymino_bot_weighted_amp_v5_warmup_{now}.pth")
    print(f"Model saved to pymino_bot_weighted_amp_v5_warmup_{now}.pth")

if __name__ == "__main__":
    main()
