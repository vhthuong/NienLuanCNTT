import time
import os
import subprocess
import re

from moonshine_voice import get_model_for_language
from moonshine_voice.transcriber import Transcriber
from moonshine_voice.utils import load_wav_file


# ========================
# CONFIG
# ========================

FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"

test_data = [
    {
        "file": "test-assets/tieng_anh.mp3",
        "language": "en",
        "ground_truth": "Mathematics is a fundamental discipline that plays a crucial role in understanding the world around us. It provides a logical framework for solving problems, analyzing patterns, and making precise calculations. From basic arithmetic to advanced theories, mathematics is used in many fields such as science, engineering, economics, and technology. It helps develop critical thinking and reasoning skills, enabling people to approach complex situations in a structured way. Despite being challenging at times, mathematics is essential for innovation and progress in modern society.",
    },
    {
        "file": "test-assets/tieng_viet.mp3",
        "language": "vi",
        "ground_truth": "Trí tuệ nhân tạo (AI) đang ngày càng trở thành một phần quan trọng trong cuộc sống hiện đại. AI cho phép máy móc học hỏi từ dữ liệu, nhận diện hình ảnh, xử lý ngôn ngữ và đưa ra quyết định một cách thông minh. Nhờ đó, AI được ứng dụng rộng rãi trong nhiều lĩnh vực như y tế, giáo dục, thương mại điện tử và giao thông. Tuy nhiên, bên cạnh những lợi ích to lớn, AI cũng đặt ra nhiều thách thức về đạo đức, bảo mật và việc làm. Vì vậy, việc phát triển và sử dụng AI một cách có trách nhiệm là điều rất cần thiết trong tương lai.",
    },
    {
        "file": "test-assets/tieng_trung.mp3",
        "language": "zh",
        "ground_truth": "化学是一门研究物质组成、结构、性质及其变化规律的科学。通过化学，人们可以了解各种物质是如何形成和相互作用的。化学在日常生活中具有重要作用，例如药品的制造、食品加工以及环境保护等领域都离不开化学知识。学习化学不仅可以提高我们的科学素养，还能帮助我们更好地理解世界的本质。",
    },
    {
        "file": "test-assets/tieng_nhat.mp3",
        "language": "ja",
        "ground_truth": "物理学は、自然界の法則や現象を研究する学問です。力、エネルギー、運動、光などの基本的な概念を通して、私たちは世界の仕組みを理解することができます。物理学は工学や技術の発展にも大きく貢献しており、日常生活のさまざまな場面で応用されています。物理を学ぶことで、論理的思考力を高め、物事を深く考える力を身につけることができます。",
    },
    {
        "file": "test-assets/tieng_han.mp3",
        "language": "ko",
        "ground_truth": "영화관은 많은 사람들이 영화를 즐기기 위해 찾는 인기 있는 장소입니다. 큰 스크린과 생생한 음향 시스템 덕분에 집에서는 느끼기 어려운 몰입감을 경험할 수 있습니다. 특히 최신 영화를 가장 먼저 볼 수 있다는 점에서 영화관은 여전히 매력적인 문화 공간으로 자리 잡고 있습니다. 친구나 가족과 함께 영화를 보며 시간을 보내는 것은 좋은 추억을 만드는 방법이기도 합니다. 또한 다양한 장르의 영화가 상영되어 각자의 취향에 맞는 작품을 선택할 수 있습니다. 최근에는 편안한 좌석과 다양한 서비스가 제공되면서 영화관의 환경도 점점 더 개선되고 있습니다.",
    },
]

# ========================
# TEXT NORMALIZATION
# ========================


def normalize_text(text):
    text = str(text).lower()

    # XÓA timestamp kiểu [000s], [12.5s]
    text = re.sub(r"\[\d+(\.\d+)?s\]", "", text)

    # bỏ dấu cơ bản (giữ unicode)
    text = re.sub(r"[.,!?()\-]", "", text)

    # xóa khoảng trắng dư
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ========================
# TOKENIZE (FIX LỖI CHÍNH)
# ========================


def tokenize(text, lang):
    if lang in ["zh", "ja", "ko"]:
        return list(text)  # tách từng ký tự
    return text.split()


# ========================
# MP3 -> WAV
# ========================


def convert_to_wav(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File không tồn tại: {input_path}")

    if input_path.endswith(".wav"):
        return input_path

    output_path = input_path.replace(".mp3", "_converted.wav")

    result = subprocess.run(
        [
            FFMPEG_PATH,
            "-y",
            "-i",
            input_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            output_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError("Lỗi khi convert MP3 -> WAV")

    return output_path


# ========================
# WER FUNCTION (FIX)
# ========================


def compute_wer(reference, hypothesis, lang):
    ref = tokenize(reference, lang)
    hyp = tokenize(hypothesis, lang)

    if len(ref) == 0:
        return 0

    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]

    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1

            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost,
            )

    wer = d[len(ref)][len(hyp)] / float(len(ref))

    return round(wer, 2)  # chặn max = 1


# ========================
# LOAD MODEL (CACHE)
# ========================

model_cache = {}


def get_transcriber(language):
    if language not in model_cache:
        model_path, model_arch = get_model_for_language(language)
        model_cache[language] = Transcriber(model_path, model_arch)
    return model_cache[language]


# ========================
# MAIN TEST
# ========================

results = []

for item in test_data:
    file_path = item["file"]
    language = item["language"]

    ground_truth = normalize_text(item["ground_truth"])

    print(f"\nTesting: {file_path} ({language})")

    try:
        transcriber = get_transcriber(language)

        wav_path = convert_to_wav(file_path)

        audio_data, sample_rate = load_wav_file(wav_path)

        start = time.time()
        transcript = transcriber.transcribe_without_streaming(audio_data, sample_rate)
        end = time.time()

        processing_time = round(end - start, 2)

        transcript_text = normalize_text(str(transcript))

        # DEBUG (rất hữu ích)
        print("GT:", ground_truth)
        print("PR:", transcript_text)

        wer = compute_wer(ground_truth, transcript_text, language)
        accuracy = round((1 - wer) * 100, 2)

        results.append(
            {
                "file": os.path.basename(file_path),
                "language": language,
                "time": processing_time,
                "wer": wer,
                "accuracy": accuracy,
            }
        )

    except Exception as e:
        print(f"Lỗi: {e}")


# ========================
# PRINT RESULT
# ========================

print("\n===== FINAL RESULT =====\n")

print(f"{'File':<20} {'Lang':<8} {'Time(s)':<10} {'WER':<8} {'Accuracy(%)':<12}")

for r in results:
    print(
        f"{r['file']:<20} {r['language']:<8} {r['time']:<10} {r['wer']:<8} {r['accuracy']:<12}"
    )


# ========================
# AVERAGE
# ========================

if results:
    avg_time = sum(r["time"] for r in results) / len(results)
    avg_wer = sum(r["wer"] for r in results) / len(results)
    avg_acc = sum(r["accuracy"] for r in results) / len(results)

    print("\n===== AVERAGE =====")
    print(f"Avg Time: {round(avg_time,2)} s")
    print(f"Avg WER: {round(avg_wer,2)}")
    print(f"Avg Accuracy: {round(avg_acc,2)} %")
