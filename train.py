import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VideoDataset, train_val_split
from model import FrameAggregatorResNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pain / no pain video classifier.")
    parser.add_argument(
        "--pain_dir",
        type=str,
        default="/Users/lucky21/Downloads/pain",
        help="Directory containing pain videos.",
    )
    parser.add_argument(
        "--no_pain_dir",
        type=str,
        default="/Users/lucky21/Downloads/no pain",
        help="Directory containing no-pain videos.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Where to save trained model.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for clips, labels in tqdm(loader, desc="Train", leave=False):
        clips = clips.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(clips)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for clips, labels in tqdm(loader, desc="Val", leave=False):
        clips = clips.to(device)
        labels = labels.to(device)

        logits = model(clips)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_samples, val_samples = train_val_split(
        pain_dir=args.pain_dir,
        no_pain_dir=args.no_pain_dir,
        val_ratio=args.val_ratio,
    )

    train_ds = VideoDataset(
        pain_dir=args.pain_dir,
        no_pain_dir=args.no_pain_dir,
        num_frames=args.num_frames,
        clips=train_samples,
    )
    val_ds = VideoDataset(
        pain_dir=args.pain_dir,
        no_pain_dir=args.no_pain_dir,
        num_frames=args.num_frames,
        clips=val_samples,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = FrameAggregatorResNet(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_path = os.path.join(args.output_dir, "pain_classifier_best.pth")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_acc": best_val_acc,
                    "epoch": epoch,
                    "num_frames": args.num_frames,
                },
                best_path,
            )
            print(f"New best model saved to {best_path} (val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    main()

