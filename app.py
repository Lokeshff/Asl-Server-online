import logging
import os
import subprocess
from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
from vosk import Model, KaldiRecognizer
import wave
import json
import uuid
from huggingface_hub import snapshot_download

# Setup logging
logging.basicConfig(
    filename='server_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# Define model names
t5_model_name = "AGLoki/asl-gloss-t5"  # T5 model directory on Hugging Face
vosk_model_name = "AGLoki/vosk-model-small-en-us-0.15"  # Vosk model directory on Hugging Face

# Set up model directories
base_dir = os.path.dirname(os.path.abspath(__file__))
t5_model_dir = os.path.join(base_dir, "t5-model")
vosk_model_dir = os.path.join(base_dir, "vosk-model-small-en-us-0.15")

# Download models using snapshot_download from Hugging Face Hub
def download_models():
    if not os.path.exists(t5_model_dir):
        logging.info(f"Downloading T5 model from {t5_model_name}...")
        snapshot_download(repo_id=t5_model_name, local_dir=t5_model_dir)

    if not os.path.exists(vosk_model_dir):
        logging.info(f"Downloading Vosk model from {vosk_model_name}...")
        snapshot_download(repo_id=vosk_model_name, local_dir=vosk_model_dir)

# Ensure models are downloaded
download_models()

# Load models
tokenizer = T5Tokenizer.from_pretrained(t5_model_dir, local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained(t5_model_dir, local_files_only=True)
vosk_model = Model(vosk_model_dir)

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

        # Generate ASL gloss (add prefix for ASL translation)
        input_text = "translate English to ASL: " + text
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids, max_length=50)[0]
        asl_gloss = tokenizer.decode(output_ids, skip_special_tokens=True)

        # Log model output
        logging.info(f"ASL Gloss Output: {asl_gloss}")

        # Cleanup raw audio file
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
