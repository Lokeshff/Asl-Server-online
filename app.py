from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
from vosk import Model as VoskModel, KaldiRecognizer
from huggingface_hub import snapshot_download
import torch
import os
import wave
import json
import subprocess
import uuid

app = Flask(__name__)

# === Step 1: Download model repo from Hugging Face ===
hf_repo_id = "AGLoki/asl-gloss-t5"
repo_dir = snapshot_download(repo_id=hf_repo_id, local_dir="models", local_dir_use_symlinks=False)

# Subfolder paths
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
    original_filename = f"temp_input_{uuid.uuid4().hex}"
    input_path = f"{original_filename}.webm"  # You can adjust extension if needed
    wav_path = f"{original_filename}.wav"

    audio_file.save(input_path)

    # === Convert to WAV (mono, PCM, 16-bit) ===
    try:
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ac", "1",               # mono
            "-ar", "16000",           # 16 kHz
            "-sample_fmt", "s16",     # 16-bit signed
            wav_path
        ]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        os.remove(input_path)
        return jsonify({"error": "Audio conversion failed. Ensure ffmpeg is installed."}), 500

    # === Speech Recognition with Vosk ===
    wf = wave.open(wav_path, "rb")
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
    os.remove(input_path)
    os.remove(wav_path)

    if not transcript.strip():
        return jsonify({"error": "No speech detected."}), 400

    # === T5 Translation to ASL Gloss ===
    input_ids = tokenizer.encode(transcript.strip(), return_tensors="pt")
    output_ids = model.generate(input_ids)
    gloss = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return jsonify({
        "transcript": transcript.strip(),
        "asl_gloss": gloss
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
