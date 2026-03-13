import os
import subprocess
import time
import wave
import re

from flask import Flask, request, render_template_string

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
# HTML
# ========================

HTML = """
<!doctype html>
<html>

<head>

<title>Moonshine Speech-to-Text</title>

<style>

body{
font-family:Arial;
background:#667eea;
display:flex;
justify-content:center;
align-items:center;
height:100vh;
}

.container{
background:white;
padding:40px;
border-radius:12px;
width:700px;
box-shadow:0 10px 25px rgba(0,0,0,0.2);
}

h2{
text-align:center;
margin-bottom:20px;
}

form{
display:flex;
flex-direction:column;
gap:12px;
}

/* SELECT LANGUAGE */

select{
width:220px;
padding:10px;
border-radius:8px;
border:1px solid #ccc;
font-size:14px;
background:#f8f8f8;
cursor:pointer;
transition:0.3s;
}

select:hover{
border-color:#667eea;
}


/* FILE UPLOAD */

input[type="file"]{
font-size:14px;
}

input[type="file"]::file-selector-button{

background:#667eea;
color:white;
border:none;
padding:8px 15px;
border-radius:6px;
cursor:pointer;
margin-right:10px;
transition:0.3s;

}

input[type="file"]::file-selector-button:hover{

background:#5563c1;

}

.file-upload{
    display:flex;
    align-items:center;
    gap:15px;
    margin:15px 0;
}

.upload-btn{
    background:#667eea;
    color:white;
    padding:10px 22px;
    border-radius:8px;
    cursor:pointer;
    font-weight:500;
    transition:0.2s;
}

.upload-btn:hover{
    background:#5563c1;
}

#fileInput{
    display:none;
}

#fileName{
    color:#333;
    font-size:14px;
}


/* BUTTON */

button{
background:#667eea;
color:white;
border:none;
padding:10px;
border-radius:6px;
cursor:pointer;
width:150px;
}

button:hover{
background:#5563c1;
}


/* INFO */

.info{
margin-top:10px;
}


/* TRANSCRIPT */

.transcript{
margin-top:20px;
background:#f4f4f4;
padding:15px;
border-radius:8px;
}

.transcript h3{
margin-top:0;
margin-bottom:10px;
}

.transcript-content{
white-space:pre-wrap;
font-family:monospace;
line-height:1.6;

max-height:300px;
overflow-y:auto;

background:white;
padding:10px;
border-radius:6px;
border-left:4px solid #667eea;
}

/* LOADING */

.loading{
margin-top:10px;
color:#667eea;
font-weight:bold;
}

</style>


<script>
function showLoading(){
document.getElementById("loading").style.display="block";
}

document.addEventListener("DOMContentLoaded", function(){

    const fileInput = document.getElementById("fileInput");
    const fileName = document.getElementById("fileName");

    fileInput.addEventListener("change", function(){

        if(fileInput.files.length > 0){
            fileName.textContent = fileInput.files[0].name;
        } 
        else{
            fileName.textContent = "No file chosen";
        }

    });

});
</script>

</head>


<body>

<div class="container">

<h2>Moonshine Speech-to-Text</h2>

<form method="post" enctype="multipart/form-data" onsubmit="showLoading()">

<label>Select Language</label>

<select name="language">

{% for code,name in languages.items() %}

<option value="{{code}}">{{name}}</option>

{% endfor %}

</select>

<div class="file-upload">

<label for="fileInput" class="upload-btn">
Choose File
</label>

<input id="fileInput" type="file" name="file" required>

<span id="fileName">No file chosen</span>

</div>

<button type="submit">Transcribe</button>

</form>

<div id="loading" class="loading" style="display:none">
Processing audio...
</div>


{% if selected_language %}

<div class="info">
<b>Language:</b> {{selected_language}}
</div>

{% endif %}


{% if uploaded_filename %}

<div class="info">
<b>File:</b> {{uploaded_filename}}
</div>

{% endif %}


{% if audio_length %}

<div class="info">
<b>Audio length:</b> {{audio_length}} seconds
</div>

{% endif %}


{% if processing_time %}

<div class="info">
<b>Processing time:</b> {{processing_time}} seconds
</div>

{% endif %}


{% if result %}

<div class="transcript">

<h3>Transcript</h3>

<div class="transcript-content">
{{result|safe}}
</div>

</div>

{% endif %}

</div>

</body>

</html>
"""


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

    return render_template_string(
        HTML,
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
