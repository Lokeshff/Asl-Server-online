import logging
import os
import uuid
import wave
import json
import subprocess
from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
from vosk import Model as VoskModel, KaldiRecognizer
from huggingface_hub import snapshot_download

# Setup logging
logging.basicConfig(
    filename='server_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# Define model source
HF_REPO_ID = "AGLoki/asl-gloss-t5"

# Base path and model paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_root_dir = os.path.join(base_dir, "asl-models")
t5_model_dir = os.path.join(model_root_dir, "t5-small")
vosk_model_dir = os.path.join(model_root_dir, "vosk-model-small-en-us-0.15")

# Download models once
if not os.path.exists(model_root_dir):
    logging.info("Downloading models from Hugging Face...")
    snapshot_download(repo_id=HF_REPO_ID, local_dir=model_root_dir, repo_type="model")

# Load models
tokenizer = T5Tokenizer.from_pretrained(t5_model_dir, local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained(t5_model_dir, local_files_only=True)
vosk_model = VoskModel(vosk_model_dir)

# Ensure folder for storing converted WAV files
os.makedirs("converted_audio", exist_ok=True)

@app.route('/')
def index():
    return "ASL Flask server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        audio_file = request.files['audio']
        raw_path = f"temp_{uuid.uuid4().hex}.input"
        wav_filename = f"{uuid.uuid4().hex}.wav"
        wav_path = os.path.join("converted_audio", wav_filename)

        # Save uploaded audio
        audio_file.save(raw_path)

        # Convert to WAV format (mono, 16kHz, 16-bit PCM)
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_path,
            "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
            wav_path
        ], check=True)

        # Speech-to-text with Vosk
        wf = wave.open(wav_path, "rb")
        rec = KaldiRecognizer(vosk_model, wf.getframerate())
        text = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                text += res.get("text", "") + " "
        res = json.loads(rec.FinalResult())
        text += res.get("text", "")
        wf.close()

        logging.info(f"Final Transcription: {text}")

        # Translate to ASL gloss using T5
        input_text = "translate English to ASL: " + text
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids, max_length=50)[0]
        asl_gloss = tokenizer.decode(output_ids, skip_special_tokens=True)

        logging.info(f"ASL Gloss Output: {asl_gloss}")

        # Cleanup temp file
        os.remove(raw_path)

        return jsonify({
            "asl_gloss": asl_gloss,
            "gesture_video": asl_gloss.upper().replace(" ", "_") + ".mp4",
            "converted_wav": wav_filename
        })

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
