from fastapi import FastAPI, HTTPException, File, UploadFile
import torch
import torch.nn as nn
from contextlib import asynccontextmanager
import numpy as np
import torchaudio.transforms as T
from model import AudioCNN
from http import HTTPStatus
import io
import soundfile as sf
import librosa

model = None
device = None
classes = None
audio_processor = None


class AudioProcessor:
    def __init__(self):
        self.transform = nn.Sequential(
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

    def process_audio_chunk(self, audio_data: np.array):
        waveform = torch.from_numpy(audio_data).float()
        waveform = waveform.unsqueeze(0)
        spectrogram = self.transform(waveform)

        return spectrogram.unsqueeze(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device, classes, audio_processor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        checkpoint = torch.load("models/best_model.pth", map_location=device)
        classes = checkpoint["classes"]
        state_dict = checkpoint["model_state_dict"]

        model = AudioCNN(len(classes))
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        print("Modelo carregado!")

        audio_processor = AudioProcessor()

    except FileNotFoundError:
        print('Erro: Arquivo "best_model.pth" não foi encontrado')

    yield

    print("Servidor desligado")


app = FastAPI(title="cnn-audio", lifespan=lifespan)


@app.post("/predict", status_code=HTTPStatus.OK)
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Modelo não foi carregado.",
        )

    if not file.filename.endswith(".wav"):
        raise HTTPException(HTTPStatus.BAD_REQUEST, detail="Envie um arquivo .wav.")

    try:
        audio_bytes = await file.read()
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        if sample_rate != 44100:
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=44100
            )

        spectrogram = audio_processor.process_audio_chunk(audio_data)
        spectrogram = spectrogram.to(device)

        with torch.no_grad():
            output = model(spectrogram)

            output = torch.nan_to_num(output)
            probs = torch.softmax(output, dim=1)[0]
            top3_prob, top3_indices = torch.topk(probs, 3)

            predictions = []

            for prob, idx in zip(top3_prob, top3_indices):
                predictions.append(
                    {"class": classes[idx], "confidence": round(prob.item(), 4)}
                )

            return {"predictions": predictions, "filename": file.filename}

    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Erro ao processar áudio: {str(e)}",
        )
