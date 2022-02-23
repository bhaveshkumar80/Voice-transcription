import torch
import librosa
import soundfile as sf
import numpy as np
from scipy.io import wavfile
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

tokenizer.save_pretrained("models/")
model.save_pretrained("models/")

@app.post("/")
async def Voice_Translation(audio_file: UploadFile = File(...)):

    input_audio, _ = librosa.load(audio_file.file, sr=16000)

    input_values = tokenizer(input_audio, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return {'text': transcription}