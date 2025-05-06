import os
import json
import uuid
import wave
import logging
import subprocess
from flask import Flask, request, jsonify
from vosk import Model as VoskModel, KaldiRecognizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from huggingface_hub import snapshot_download

# Setup logging
logging.basicConfig(filename="server_log.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

# Create directories
os.makedirs("converted_audio", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ------------------ Download and Load T5 Model ------------------
print("Downloading T5 model from Hugging Face...")
snapshot_download(repo_id="AGLoki/asl-gloss-t5")  # Just ensure it's downloaded

# Load tokenizer and model using repo_id instead of local path
tokenizer = T5Tokenizer.from_pretrained("AGLoki/asl-gloss-t5")
t5_model = T5ForConditionalGeneration.from_pretrained("AGLoki/asl-gloss-t5")

# ------------------ Download and Load Vosk Model ------------------
vosk_model_path = "models/vosk"
if not os.path.exists(os.path.join(vosk_model_path, "conf")):
    print("Downloading Vosk model from Hugging Face...")
    snapshot_download(repo_id="AGLoki/vosk-model-small-en-us-0.15", local_dir=vosk_model_path)

vosk_model = VoskModel(vosk_model_path)

@app.route('/')
def index():
    return "ASL Gloss Server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        audio_file = request.files['audio']
        raw_input_path = f"temp_{uuid.uuid4().hex}.input"
        wav_filename = f"{uuid.uuid4().hex}.wav"
        wav_path = os.path.join("converted_audio", wav_filename)

        # Save incoming file
        audio_file.save(raw_input_path)

        # Convert to mono 16kHz WAV using ffmpeg
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_input_path,
            "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
            wav_path
        ], check=True)

        # Transcribe using Vosk
        wf = wave.open(wav_path, "rb")
        recognizer = KaldiRecognizer(vosk_model, wf.getframerate())
        text = ""
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text += result.get("text", "") + " "
        final_result = json.loads(recognizer.FinalResult())
        text += final_result.get("text", "")
        wf.close()

        logging.info(f"Transcribed Text: {text.strip()}")

        # Translate to ASL Gloss
        input_text = "translate English to ASL: " + text.strip()
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        output_ids = t5_model.generate(input_ids, max_length=50)[0]
        gloss = tokenizer.decode(output_ids, skip_special_tokens=True)

        logging.info(f"ASL Gloss: {gloss}")

        os.remove(raw_input_path)

        return jsonify({
            "asl_gloss": gloss,
            "gesture_video": gloss.upper().replace(" ", "_") + ".mp4"
        })

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
