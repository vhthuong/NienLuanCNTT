from moonshine_voice import get_model_for_language
from moonshine_voice.transcriber import Transcriber
from moonshine_voice.utils import load_wav_file

print("🔄 Loading model...")

model_path, model_arch = get_model_for_language("vi")
transcriber = Transcriber(model_path, model_arch)

print("📂 Loading audio file...")
audio_data, sample_rate = load_wav_file("test-assets/two_cities_16k.wav")

print("🧠 Transcribing...")
result = transcriber.transcribe_without_streaming(audio_data, sample_rate)

print("\n===== RESULT =====")
print(result)

transcriber.close()
