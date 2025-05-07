import logging
from flask import Flask, request, jsonify
import os
import torch
import subprocess
from transformers import T5ForConditionalGeneration, T5Tokenizer
from vosk import Model as VoskModel, KaldiRecognizer
from huggingface_hub import snapshot_download
import wave
import json
import uuid

# Setup logging
logging.basicConfig(
    filename='server_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger().addHandler(logging.StreamHandler())  # Also log to console

app = Flask(__name__)

# === Step 1: Download models from Hugging Face ===
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

# Ensure folder for storing converted WAV files
os.makedirs("converted_audio", exist_ok=True)

@app.route('/')
def index():
    return "Flask server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        audio_file = request.files['audio']
        raw_path = f"temp_{uuid.uuid4().hex}.input"
        wav_filename = f"{uuid.uuid4().hex}.wav"
        wav_path = os.path.join("converted_audio", wav_filename)

        # Save uploaded audio file
        audio_file.save(raw_path)

        # Convert to WAV (mono, 16kHz) using ffmpeg
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_path,
            "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
            wav_path
        ], check=True)

        # Transcribe with Vosk
        wf = wave.open(wav_path, "rb")
        rec = KaldiRecognizer(vosk_model, wf.getframerate())
        text = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                text += res.get("text", "")
                logging.info(f"Partial Transcription: {res.get('text', '')}")
        res = json.loads(rec.FinalResult())
        text += res.get("text", "")
        wf.close()

        logging.info(f"Final Transcription: {text}")

        # Generate ASL gloss
        input_text = "translate English to ASL: " + text
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids, max_length=50)[0]
        asl_gloss = tokenizer.decode(output_ids, skip_special_tokens=True)

        logging.info(f"ASL Gloss Output: {asl_gloss}")

        os.remove(raw_path)

        return jsonify({
            "asl_gloss": asl_gloss,
            "gesture_video": asl_gloss.upper().replace(" ", "_") + ".mp4",
            "converted_wav": wav_filename
        })

    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
