#  CNN Audio Classifier

A ResNet-inspired Convolutional Neural Network for environmental sound classification, trained on the **ESC-50** dataset. Exposes predictions via a **FastAPI** REST endpoint — send a `.wav` file and get back the top-3 most likely sound classes with confidence scores.

---

## Project Structure

```
cnn-audio-classifier/
├── models/
│   └── best_model.pth        # Best checkpoint saved during training
├── model.py                  # AudioCNN architecture (ResNet-like)
├── train.py                  # Training pipeline (ESC-50 dataset)
├── app.py                    # FastAPI inference server
├── pyproject.toml            # Project metadata and dependencies
└── uv.lock                   # Locked dependency versions
```

---

## Architecture

`AudioCNN` is a ResNet-inspired CNN that operates on **mel-spectrograms**.

| Stage        | Details                                      |
|--------------|----------------------------------------------|
| Input        | Mono mel-spectrogram (1 × 128 × T)           |
| Stem         | Conv2d 7×7, stride 2 → BN → ReLU → MaxPool  |
| Layer 1      | 3 × ResidualBlock (64 channels)              |
| Layer 2      | 4 × ResidualBlock (128 channels, stride 2)   |
| Layer 3      | 6 × ResidualBlock (256 channels, stride 2)   |
| Layer 4      | 3 × ResidualBlock (512 channels, stride 2)   |
| Head         | AdaptiveAvgPool → Dropout(0.5) → Linear(512 → N classes) |

Each `ResidualBlock` follows the standard pattern: Conv → BN → ReLU → Conv → BN → residual add → ReLU, with a 1×1 projection shortcut when dimensions change.

---

## Dataset — ESC-50

The model is trained on [ESC-50](https://github.com/karolpiczak/ESC-50): a benchmark dataset of **2,000 environmental audio recordings** across **50 classes** (e.g., dog barking, rain, chainsaw, clapping).

- Folds 1–4 → training set
- Fold 5 → validation set

**Download and place the dataset at:**
```
data/ESC-50-master/
├── audio/          # .wav files
└── meta/
    └── esc50.csv   # Metadata file
```

---

## Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) package manager

## Installation

```bash
# Clone the repository
git clone https://github.com/koheiseko/cnn-audio-classifier.git
cd cnn-audio-classifier

# Install dependencies
uv sync
```

---

## Training

```bash
uv run python train.py
```

### Hyperparameters

| Parameter       | Value                          |
|-----------------|--------------------------------|
| Epochs          | 115                            |
| Batch size      | 32                             |
| Optimizer       | AdamW (lr = 0.005)             |
| LR Scheduler    | OneCycleLR (max_lr = 0.002, warmup 10%) |
| Loss            | CrossEntropyLoss               |
| Metric          | Multiclass Accuracy            |
| Sample rate     | 44,100 Hz                      |
| Spectrogram     | MelSpectrogram (n_mels=128, n_fft=1024, hop_length=512) |

### Data Augmentation (training only)

- **Frequency Masking** — masks up to 30 frequency bins
- **Time Masking** — masks up to 80 time steps

Training logs are written to `models/tensorboard_logs/` and can be visualized with TensorBoard:

```bash
uv run tensorboard --logdir models/tensorboard_logs
```

The best model (highest validation accuracy) is automatically saved to `models/best_model.pth`.

---

## Running the API

```bash
uv run fastapi dev app.py      # development
uv run fastapi run app.py      # production
```

The server loads the saved checkpoint from `models/best_model.pth` on startup.

### `POST /predict`

Send a `.wav` audio file and receive the top-3 predicted classes with confidence scores.

**Request:**

```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@your_audio.wav"
```

**Response:**

```json
{
  "predictions": [
    { "class": "dog",    "confidence": 0.8732 },
    { "class": "cat",    "confidence": 0.0651 },
    { "class": "crow",   "confidence": 0.0411 }
  ],
  "filename": "your_audio.wav"
}
```

> **Note:** Only `.wav` files are accepted. Multi-channel audio is automatically converted to mono, and audio is resampled to 44,100 Hz if needed.

---

## Dependencies

Dependencies are managed with [`uv`](https://github.com/astral-sh/uv). Key libraries:

- **PyTorch / torchaudio** — model, training, and audio transforms
- **FastAPI** — REST inference server
- **librosa / soundfile** — audio I/O and resampling
- **torchmetrics** — accuracy tracking during training
- **TensorBoard** — training visualization
- **pandas** — ESC-50 metadata parsing

---

## 👤 Author

**Kohei Seko** — [@koheiseko](https://github.com/koheiseko)