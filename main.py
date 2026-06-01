from math import gcd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import speech_recognition as srec
from scipy.signal import butter, resample_poly, sosfiltfilt


SAMPLE_RATE = 44100
SAMPLE_WIDTH = 2
DTYPE = np.int16

SOUNDS_DIR = Path("./Sounds")
NAME_ORIGINAL_WAV = f"./Sounds/Sound_{SAMPLE_RATE}[Hz]_{SAMPLE_WIDTH}[byte].wav"
NAME_ORIGINAL_RAW = f"./Sounds/Sound_{SAMPLE_RATE}[Hz]_{SAMPLE_WIDTH}[byte].raw"
NAME_RESAMPLED_WAV = "./Sounds/Sound_4000[Hz]_2[byte].wav"
NAME_RESAMPLED_RAW = "./Sounds/Sound_4000[Hz]_2[byte].raw"
NAME_FILTERED_WAV = "./Sounds/Filtered_4000[Hz]_2[byte].wav"
NAME_FILTERED_RAW = "./Sounds/Filtered_4000[Hz]_2[byte].raw"
NAME_PLOT = "./Sounds/signals_comparison.png"


def recognize_speech(rec, mic):
    with mic as source:
        rec.adjust_for_ambient_noise(source)
        print("Говоріть...")
        audio = rec.listen(source)

    result = {"Текст": None}

    try:
        result["Текст"] = rec.recognize_google(
            audio,
            show_all=False,
            language="uk-UA",
        )
    except srec.UnknownValueError:
        result["Текст"] = "Мову не розпізнано"
    except srec.RequestError as error:
        result["Текст"] = f"Помилка сервісу розпізнавання: {error}"

    return result


def save_wav_and_raw(audio):
    wav_data = audio.get_wav_data(
        convert_rate=SAMPLE_RATE,
        convert_width=SAMPLE_WIDTH,
    )
    raw_data = audio.get_raw_data(
        convert_rate=SAMPLE_RATE,
        convert_width=SAMPLE_WIDTH,
    )

    with open(NAME_ORIGINAL_WAV, "wb") as file:
        file.write(wav_data)

    with open(NAME_ORIGINAL_RAW, "wb") as file:
        file.write(raw_data)


def resample_wav_and_raw():
    fs_target = 4000

    data, fs_original = sf.read(NAME_ORIGINAL_WAV)
    if len(data.shape) > 1:
        data = data[:, 0]

    divisor = gcd(fs_original, fs_target)
    up = fs_target // divisor
    down = fs_original // divisor
    data_resampled = resample_poly(data, up, down)
    sf.write(NAME_RESAMPLED_WAV, data_resampled, fs_target)

    with open(NAME_ORIGINAL_RAW, "rb") as file:
        raw_bytes = file.read()

    signal = np.frombuffer(raw_bytes, dtype=DTYPE)
    signal_float = signal.astype(np.float32) / 32768.0
    resampled = resample_poly(signal_float, up, down)
    resampled_int16 = np.int16(np.clip(resampled, -1.0, 1.0) * 32767)

    with open(NAME_RESAMPLED_RAW, "wb") as file:
        file.write(resampled_int16.tobytes())


def filter_wav_and_raw():
    cutoff = 4000
    order = 6

    data, _ = sf.read(NAME_ORIGINAL_WAV)
    if len(data.shape) > 1:
        data = data[:, 0]

    sos = butter(order, cutoff, btype="low", fs=SAMPLE_RATE, output="sos")
    filtered = sosfiltfilt(sos, data)
    sf.write(NAME_FILTERED_WAV, filtered, SAMPLE_RATE)

    with open(NAME_ORIGINAL_RAW, "rb") as file:
        raw_bytes = file.read()

    signal = np.frombuffer(raw_bytes, dtype=DTYPE)
    signal_float = signal.astype(np.float32) / 32768.0
    filtered_raw = sosfiltfilt(sos, signal_float)
    filtered_int16 = np.int16(np.clip(filtered_raw, -1.0, 1.0) * 32767)

    with open(NAME_FILTERED_RAW, "wb") as file:
        file.write(filtered_int16.tobytes())


def plot_signals():
    cutoff = 4000

    data_original, fs_original = sf.read(NAME_ORIGINAL_WAV)
    if len(data_original.shape) > 1:
        data_original = data_original[:, 0]
    time_original = np.arange(len(data_original)) / fs_original * 1000

    data_resampled, fs_resampled = sf.read(NAME_RESAMPLED_WAV)
    if len(data_resampled.shape) > 1:
        data_resampled = data_resampled[:, 0]
    time_resampled = np.arange(len(data_resampled)) / fs_resampled * 1000

    data_filtered, fs_filtered = sf.read(NAME_FILTERED_WAV)
    if len(data_filtered.shape) > 1:
        data_filtered = data_filtered[:, 0]
    time_filtered = np.arange(len(data_filtered)) / fs_filtered * 1000

    plt.figure(figsize=(12, 6))
    plt.plot(time_original, data_original, label=f"Оригінал (fs={SAMPLE_RATE} Гц)")
    plt.plot(time_resampled, data_resampled, label=f"Ресемпл (fs={fs_resampled} Гц)")
    plt.plot(time_filtered, data_filtered, label=f"Фільтрований (LPF {cutoff} Гц)")
    plt.title("Порівняння сигналів у часовій області")
    plt.xlabel("Час (мс)")
    plt.ylabel("Амплітуда")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(NAME_PLOT, dpi=150)
    plt.show()


def sound_recoder(rec, mic):
    SOUNDS_DIR.mkdir(exist_ok=True)

    with mic as source:
        rec.adjust_for_ambient_noise(source)
        print("Говоріть...")
        audio = rec.listen(source)

    save_wav_and_raw(audio)
    resample_wav_and_raw()
    filter_wav_and_raw()
    plot_signals()


if __name__ == "__main__":
    recognizer = srec.Recognizer()
    microphone = srec.Microphone(device_index=1, sample_rate=SAMPLE_RATE)
    sound_recoder(recognizer, microphone)
