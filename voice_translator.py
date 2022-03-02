#http://ec2-35-86-155-156.us-west-2.compute.amazonaws.com:5000/docs

import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

#model_path = "models/"
model_path = "facebook/wav2vec2-large-960h"

processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)

tokenizer.save_pretrained("models/")
model.save_pretrained("models/")

@app.post("/")
async def Voice_Translation(audio_file: UploadFile = File(...)):
    try:
        input_audio, _ = librosa.load(audio_file.file, sr=16000)

        # tokenize
        input_values = processor(input_audio, return_tensors="pt").input_values  # Batch size 1

        # retrieve logits
        logits = model(input_values).logits

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        return {'status': True, 'text': transcription}
    except:
        return {'status': False, 'text': '', 'message': 'Something Went Wrong!'}

# audio_file = "output.wav"
# text = Voice_Translation(audio_file)
# print(text)