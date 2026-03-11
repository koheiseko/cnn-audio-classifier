import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchaudio
import torchaudio.transforms as T
from torchmetrics import Accuracy

import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from model import AudioCNN


class ESC50Dataset(Dataset):
    def __init__(
        self, data_dir: Path, metadata_dir: Path, train: bool = True, transforms=None
    ):
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_dir)

        self.classes = sorted(self.metadata["category"].unique())
        self.class_to_idx = {_class: idx for idx, _class in enumerate(self.classes)}
        self.metadata["label"] = self.metadata["category"].map(self.class_to_idx)

        self.transforms = transforms
        self.train = train

        if self.train:
            self.metadata = self.metadata[self.metadata["fold"] != 5]
        else:
            self.metadata = self.metadata[self.metadata["fold"] == 5]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = row["filename"]
        audio_path = self.data_dir / "audio" / filename

        waveform, _ = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform(dim=0, keepdim=True))

        if self.transforms:
            spectogram = self.transforms(waveform)
        else:
            spectogram = waveform

        return spectogram, row["label"]


def evaluate(model, data_loader, criterion, metric, device):
    model.eval()
    metric.reset()

    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss

            metric.update(y_pred, y_batch)

    mean_loss = total_loss / len(data_loader)

    return mean_loss, metric


def train(
    model,
    optimizer,
    scheduler,
    criterion,
    metric,
    train_loader,
    valid_loader,
    n_epochs,
    writer,
    classes,
    device,
):
    best_metric = 0.0

    for epoch in range(n_epochs):
        metric.reset()
        model.train()

        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}")

        for X_batch, y_batch in progress_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch)
            total_loss += loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            metric.update(y_pred, y_batch)

            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        train_mean_loss = (total_loss / len(train_loader)).item()
        train_metric = metric.compute().item()
        valid_mean_loss, valid_metric = evaluate(
            model, valid_loader, criterion, metric, device
        )
        valid_mean_loss, valid_metric = (
            valid_mean_loss.item(),
            valid_metric.compute().item(),
        )

        writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Loss/Train", train_mean_loss, epoch)
        writer.add_scalar("Metric/Train", train_metric, epoch)
        writer.add_scalar("Loss/Val", valid_mean_loss, epoch)
        writer.add_scalar("Metric/Val", valid_metric, epoch)

        print(
            f"Epoch {epoch}/{n_epochs}, Train Loss: {train_mean_loss:.4f}, Valid Loss: {valid_mean_loss:.4f}, Train Metric: {train_metric:.4f}, Valid Metric: {valid_metric:.4f}"
        )

        if valid_metric > best_metric:
            print(f"[INFO] New best model saved, metric: {valid_metric:.4f}")

            best_metric = valid_metric
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "accuracy": valid_metric,
                    "epoch": epoch,
                    "classes": classes,
                },
                "models/best_model.pth",
            )

    writer.close()

    print(f"[INFO] Training completed. Best metric: {best_metric:.4f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"models/tensorboard_logs/run_{timestamp}"
    writer = SummaryWriter(log_dir)

    data_dir = Path("data/ESC-50-master")
    metadata_dir = data_dir / "meta" / "esc50.csv"

    train_transforms = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=44100,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=22050,
        ),
        T.AmplitudeToDB(),
        T.FrequencyMasking(30),
        T.TimeMasking(80),
    )

    val_transforms = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=44100,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=22050,
        ),
        T.AmplitudeToDB(),
    )

    train_dataset = ESC50Dataset(
        data_dir=data_dir,
        metadata_dir=metadata_dir,
        train=True,
        transforms=train_transforms,
    )
    val_dataset = ESC50Dataset(
        data_dir=data_dir,
        metadata_dir=metadata_dir,
        train=False,
        transforms=val_transforms,
    )

    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size)

    n_classes = len(train_dataset.classes)
    model = AudioCNN(n_classes=n_classes).to(device)

    lr = 0.005
    n_epochs = 115
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    xentropy = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=0.002,
        epochs=n_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )
    accuracy = Accuracy("multiclass", num_classes=n_classes).to(device)

    print(f"[INFO] Training started for {n_epochs} epochs")

    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=xentropy,
        metric=accuracy,
        train_loader=train_loader,
        valid_loader=valid_loader,
        n_epochs=n_epochs,
        writer=writer,
        classes=train_dataset.classes,
        device=device,
    )


if __name__ == "__main__":
    main()
