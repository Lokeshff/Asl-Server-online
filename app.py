from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
from vosk import Model as VoskModel, KaldiRecognizer
from huggingface_hub import snapshot_download
import torch
import os
import wave
import json

app = Flask(__name__)

# === Step 1: Download full repo (non-zipped) from Hugging Face ===
hf_repo_id = "AGLoki/asl-gloss-t5"
repo_dir = snapshot_download(repo_id=hf_repo_id, local_dir="models", local_dir_use_symlinks=False)

# Paths to subfolders
t5_path = os.path.join(repo_dir, "t5-small")
vosk_path = os.path.join(repo_dir, "vosk-model-small-en-us-0.15")

# === Step 2: Load models ===
print("Loading T5 model from:", t5_path)
tokenizer = T5Tokenizer.from_pretrained(t5_path)
model = T5ForConditionalGeneration.from_pretrained(t5_path)

print("Loading Vosk model from:", vosk_path)
vosk_model = VoskModel(vosk_path)

# === Step 3: API Route ===
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio']
    audio_path = "temp.wav"
    audio_file.save(audio_path)

    # Process audio with Vosk
    wf = wave.open(audio_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        return jsonify({"error": "Audio must be WAV format mono PCM."}), 400

    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    rec.SetWords(True)

    transcript = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            transcript += result.get("text", "") + " "

    wf.close()
    os.remove(audio_path)

    if not transcript.strip():
        return jsonify({"error": "No speech detected."}), 400

    # Generate gloss using T5
    input_ids = tokenizer.encode(transcript.strip(), return_tensors="pt")
    output_ids = model.generate(input_ids)
    gloss = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return jsonify({
        "transcript": transcript.strip(),
        "asl_gloss": gloss
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
