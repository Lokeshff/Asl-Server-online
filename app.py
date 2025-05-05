import os
import uuid
import json
import wave
import subprocess
from flask import Flask, request, jsonify
from vosk import Model as VoskModel, KaldiRecognizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)

# Paths to models
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"
T5_MODEL_PATH = "models/t5-small"

# Load models once
vosk_model = VoskModel(VOSK_MODEL_PATH)
tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_PATH)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_PATH)

# Ensure audio output folder
os.makedirs("converted_audio", exist_ok=True)

@app.route("/")
def home():
    return "Render ASL Translator Server Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        audio = request.files['audio']
        raw_filename = f"temp_{uuid.uuid4().hex}.input"
        wav_filename = f"{uuid.uuid4().hex}.wav"
        wav_path = os.path.join("converted_audio", wav_filename)

        # Save raw audio
        audio.save(raw_filename)

        # Convert to mono 16kHz WAV
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_filename,
            "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
            wav_path
        ], check=True)

        # Transcribe with Vosk
        wf = wave.open(wav_path, "rb")
        rec = KaldiRecognizer(vosk_model, wf.getframerate())

        text = ""
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text += result.get("text", "")
        final_result = json.loads(rec.FinalResult())
        text += final_result.get("text", "")
        wf.close()

        # Generate ASL gloss from transcription
        input_text = "translate English to ASL: " + text
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        output_ids = t5_model.generate(input_ids, max_length=50)
        gloss = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return jsonify({
            "text": text,
            "asl_gloss": gloss,
            "gesture_video": gloss.upper().replace(" ", "_") + ".mp4"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
