import os
import subprocess
import time
import wave
import re

from flask import Flask, request, render_template

from moonshine_voice import get_model_for_language
from moonshine_voice.transcriber import Transcriber
from moonshine_voice.utils import load_wav_file


# ========================
# CONFIG
# ========================

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"


# ========================
# LANGUAGES
# ========================

languages = {
    "ar": "Arabic",
    "es": "Spanish",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "vi": "Vietnamese",
    "uk": "Ukrainian",
    "zh": "Chinese",
}


# ========================
# INIT APP
# ========================

app = Flask(__name__)


# ========================
# AUDIO LENGTH
# ========================


def get_audio_length(path):

    with wave.open(path, "r") as f:

        frames = f.getnframes()
        rate = f.getframerate()

        duration = frames / float(rate)

    return round(duration, 2)


# ========================
# FORMAT TRANSCRIPT
# ========================


def format_transcript(text):

    text = str(text)

    timestamps = re.findall(r"\[(\d+\.\d+)s\]", text)
    segments = re.split(r"\[\d+\.\d+s\]", text)

    lines = []

    for i in range(1, len(segments)):

        start = timestamps[i - 1]

        if i < len(timestamps):
            end = timestamps[i]
        else:
            end = "END"

        content = segments[i].strip()

        if content:
            lines.append(f"[{start}s → {end}s] {content}")

    return "<br><br>".join(lines)


# ========================
# ROUTE
# ========================


@app.route("/", methods=["GET", "POST"])
def upload():

    result = None
    uploaded_filename = None
    processing_time = None
    audio_length = None
    selected_language = None

    if request.method == "POST":

        language = request.form["language"]
        selected_language = languages.get(language)

        print("Language:", language)

        model_path, model_arch = get_model_for_language(language)
        transcriber = Transcriber(model_path, model_arch)

        file = request.files["file"]

        uploaded_filename = file.filename

        filepath = os.path.join(UPLOAD_FOLDER, uploaded_filename)

        file.save(filepath)

        if uploaded_filename.lower().endswith(".mp3"):

            wav_path = filepath.replace(".mp3", "_converted.wav")

            subprocess.run(
                [
                    FFMPEG_PATH,
                    "-y",
                    "-i",
                    filepath,
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-sample_fmt",
                    "s16",
                    wav_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            filepath = wav_path

        audio_length = get_audio_length(filepath)

        audio_data, sample_rate = load_wav_file(filepath)

        start = time.time()

        transcript = transcriber.transcribe_without_streaming(audio_data, sample_rate)

        end = time.time()

        processing_time = round(end - start, 2)

        result = format_transcript(transcript)

    return render_template(
        "index.html",
        result=result,
        uploaded_filename=uploaded_filename,
        processing_time=processing_time,
        audio_length=audio_length,
        selected_language=selected_language,
        languages=languages,
    )


# ========================
# RUN
# ========================

if __name__ == "__main__":

    app.run(debug=False)
