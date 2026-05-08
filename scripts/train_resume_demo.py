#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.checkpointing import (
    CheckpointBundle,
    append_metrics_row,
    ensure_run_dirs,
    load_checkpoint,
    save_checkpoint,
    save_run_state,
)


def make_demo_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(42)

    x_train = torch.randn(4096, 128, generator=g)
    y_train = (x_train.sum(dim=1) > 0).long()
    x_val = torch.randn(1024, 128, generator=g)
    y_val = (x_val.sum(dim=1) > 0).long()

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable training demo with last/best checkpoints")
    parser.add_argument("--run-dir", default="data/artifacts/runs/demo_classifier_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    run_dir, ckpt_dir, logs_dir = ensure_run_dirs(Path(args.run_dir).resolve())
    train_loader, val_loader = make_demo_loaders(batch_size=args.batch_size)

    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 2),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    global_step = 0
    best_val_acc = 0.0

    last_ckpt_path = ckpt_dir / "last.pt"

    if args.resume and last_ckpt_path.exists():
        checkpoint = load_checkpoint(last_ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val_acc = float(checkpoint.get("best_metric", 0.0))
        print(f"Resumed from epoch={start_epoch}, best_val_acc={best_val_acc:.4f}")

    interrupted = {"flag": False}

    def handle_stop(signum, _frame):
        interrupted["flag"] = True
        print(f"\nReceived signal {signum}. Will checkpoint and stop after this epoch.")

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_n = y.size(0)
            running_loss += loss.item() * batch_n
            seen += batch_n
            global_step += 1

        train_loss = running_loss / max(seen, 1)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        metrics = {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        if val_acc >= best_val_acc:
            best_val_acc = val_acc

        bundle = CheckpointBundle(
            epoch=epoch,
            global_step=global_step,
            best_metric=best_val_acc,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
        )

        save_checkpoint(ckpt_dir=ckpt_dir, bundle=bundle, metrics=metrics, file_name="last.pt")

        if val_acc >= best_val_acc:
            save_checkpoint(ckpt_dir=ckpt_dir, bundle=bundle, metrics=metrics, file_name="best.pt")

        append_metrics_row(
            logs_dir=logs_dir,
            row={
                "epoch": epoch,
                "global_step": global_step,
                **metrics,
                "best_val_acc": best_val_acc,
            },
        )

        save_run_state(
            run_dir=run_dir,
            payload={
                "status": "running",
                "epoch": epoch,
                "global_step": global_step,
                "best_val_acc": best_val_acc,
                "latest_metrics": metrics,
                "checkpoint_last": str(last_ckpt_path),
                "checkpoint_best": str(ckpt_dir / "best.pt"),
            },
        )

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} best={best_val_acc:.4f}"
        )

        if interrupted["flag"]:
            save_run_state(
                run_dir=run_dir,
                payload={
                    "status": "interrupted",
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_acc": best_val_acc,
                    "latest_metrics": metrics,
                    "checkpoint_last": str(last_ckpt_path),
                    "checkpoint_best": str(ckpt_dir / "best.pt"),
                },
            )
            print("Stopped safely. Resume with --resume")
            return 0

    save_run_state(
        run_dir=run_dir,
        payload={
            "status": "completed",
            "epoch": args.epochs - 1,
            "global_step": global_step,
            "best_val_acc": best_val_acc,
            "checkpoint_last": str(last_ckpt_path),
            "checkpoint_best": str(ckpt_dir / "best.pt"),
        },
    )
    print("Training completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
