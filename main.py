import glob
from math import gcd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt
import soundfile as sf
import speech_recognition as srec
from scipy.signal import butter, convolve, resample, resample_poly, sosfiltfilt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skimage.restoration import (
    cycle_spin,
    denoise_bilateral,
    denoise_invariant,
    denoise_tv_chambolle,
    denoise_wavelet,
)


SAMPLE_RATE = 44100
SAMPLE_WIDTH = 2
DTYPE = np.int16

BASE_DIR = Path(__file__).resolve().parent
SOUNDS_DIR = BASE_DIR / "Sounds"
PLOTS_DIR = SOUNDS_DIR

NAME_ORIGINAL_WAV = SOUNDS_DIR / f"Sound_{SAMPLE_RATE}[Hz]_{SAMPLE_WIDTH}[byte].wav"
NAME_ORIGINAL_RAW = SOUNDS_DIR / f"Sound_{SAMPLE_RATE}[Hz]_{SAMPLE_WIDTH}[byte].raw"
NAME_RESAMPLED_WAV = SOUNDS_DIR / "Sound_4000[Hz]_2[byte].wav"
NAME_RESAMPLED_RAW = SOUNDS_DIR / "Sound_4000[Hz]_2[byte].raw"
NAME_FILTERED_WAV = SOUNDS_DIR / "Filtered_4000[Hz]_2[byte].wav"
NAME_FILTERED_RAW = SOUNDS_DIR / "Filtered_4000[Hz]_2[byte].raw"

TARGET_RATE = 4000
FILTER_CUTOFF = 4000
FILTER_ORDER = 6
MICROPHONE_INDEX = 1


def ensure_directories():
    SOUNDS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)


def to_mono(data):
    if len(data.shape) > 1:
        return data[:, 0]
    return data


def save_recording(audio):
    wav_data = audio.get_wav_data(
        convert_rate=SAMPLE_RATE,
        convert_width=SAMPLE_WIDTH,
    )
    raw_data = audio.get_raw_data(
        convert_rate=SAMPLE_RATE,
        convert_width=SAMPLE_WIDTH,
    )

    with open(NAME_ORIGINAL_WAV, "wb") as f:
        f.write(wav_data)

    with open(NAME_ORIGINAL_RAW, "wb") as f:
        f.write(raw_data)


def resample_wav():
    data, fs_original = sf.read(NAME_ORIGINAL_WAV)
    data = to_mono(data)

    g = gcd(fs_original, TARGET_RATE)
    up = TARGET_RATE // g
    down = fs_original // g
    data_resampled = resample_poly(data, up, down)

    sf.write(NAME_RESAMPLED_WAV, data_resampled, TARGET_RATE)


def resample_raw():
    with open(NAME_ORIGINAL_RAW, "rb") as f:
        raw_bytes = f.read()

    signal = np.frombuffer(raw_bytes, dtype=DTYPE)
    signal_float = signal.astype(np.float32) / 32768.0

    g = gcd(SAMPLE_RATE, TARGET_RATE)
    up = TARGET_RATE // g
    down = SAMPLE_RATE // g
    resampled = resample_poly(signal_float, up, down)
    resampled_int16 = np.int16(np.clip(resampled, -1.0, 1.0) * 32767)

    with open(NAME_RESAMPLED_RAW, "wb") as f:
        f.write(resampled_int16.tobytes())


def filter_wav():
    data, fs_original = sf.read(NAME_ORIGINAL_WAV)
    data = to_mono(data)

    sos = butter(
        FILTER_ORDER,
        FILTER_CUTOFF,
        btype="low",
        fs=fs_original,
        output="sos",
    )
    filtered = sosfiltfilt(sos, data)

    sf.write(NAME_FILTERED_WAV, filtered, fs_original)


def filter_raw():
    with open(NAME_ORIGINAL_RAW, "rb") as f:
        raw_bytes = f.read()

    signal = np.frombuffer(raw_bytes, dtype=DTYPE)
    signal_float = signal.astype(np.float32) / 32768.0

    sos = butter(
        FILTER_ORDER,
        FILTER_CUTOFF,
        btype="low",
        fs=SAMPLE_RATE,
        output="sos",
    )
    filtered_raw = sosfiltfilt(sos, signal_float)
    filtered_int16 = np.int16(np.clip(filtered_raw, -1.0, 1.0) * 32767)

    with open(NAME_FILTERED_RAW, "wb") as f:
        f.write(filtered_int16.tobytes())


def plot_signals():
    original, fs_original = sf.read(NAME_ORIGINAL_WAV)
    resampled, fs_resampled = sf.read(NAME_RESAMPLED_WAV)
    filtered, fs_filtered = sf.read(NAME_FILTERED_WAV)

    original = to_mono(original)
    resampled = to_mono(resampled)
    filtered = to_mono(filtered)

    time_original = np.arange(len(original)) / fs_original
    time_resampled = np.arange(len(resampled)) / fs_resampled
    time_filtered = np.arange(len(filtered)) / fs_filtered

    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(time_original, original, label="Original")
    plt.title("Original signal")
    plt.xlabel("Time, s")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_resampled, resampled, label="Resampled")
    plt.title("Resampled signal")
    plt.xlabel("Time, s")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_filtered, filtered, label="Filtered")
    plt.title("Filtered signal")
    plt.xlabel("Time, s")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "Practical_2_comparison.png", dpi=150)
    plt.close()


def process_recording():
    ensure_directories()
    resample_wav()
    resample_raw()
    filter_wav()
    filter_raw()
    plot_signals()


def sound_recoder(rec, mic):
    with mic as source:
        print("Speak now...")
        audio = rec.listen(source)

    save_recording(audio)
    process_recording()


def wavelet_denoiser(signal, level=5, mode="hard", wavelet="db4"):
    """
    Denoise a 1D signal using Discrete Wavelet Transform thresholding.

    Args:
        signal (np.ndarray): The input signal.
        level (int): The level of decomposition.
        mode (str): Thresholding mode, 'soft' or 'hard'.
        wavelet (str): The name of the mother wavelet.

    Returns:
        np.ndarray: The denoised signal.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(signal.size))
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]
    ]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    return denoised_signal[: len(signal)]


def invarince_denoiser(image, **kwargs):
    return denoise_wavelet(image, sigma=0.5, wavelet="db4", mode="soft")


def gaussian_kernel(size, sigma):
    x = np.linspace(-(size // 2), size // 2, size)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def save_filtered_plot(time, data, filtered, title, output_name):
    plt.figure(figsize=(10, 6))
    plt.plot(time, data, "b-", label="Original Clean Signal")
    plt.plot(time, filtered, "g-", linewidth=2, label=title)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / output_name, dpi=150)
    plt.close()


def sound_filter():
    ensure_directories()

    data, fs_original = sf.read(NAME_ORIGINAL_WAV)
    data = to_mono(data)
    time = np.arange(len(data)) / fs_original
    data_2d = data.reshape(1, -1)

    invariance = denoise_invariant(
        data_2d,
        denoise_function=invarince_denoiser,
    ).flatten()
    total_variation = denoise_tv_chambolle(
        data_2d,
        weight=0.1,
        channel_axis=None,
    ).flatten()
    bilateral = denoise_bilateral(
        data_2d,
        sigma_color=0.05,
        sigma_spatial=15,
        channel_axis=None,
    ).flatten()
    wavelet = wavelet_denoiser(data, level=5, mode="soft", wavelet="db4")

    sf.write(SOUNDS_DIR / "Filtered_Invariance.wav", invariance, SAMPLE_RATE)
    sf.write(SOUNDS_DIR / "Filtered_Total_Variation.wav", total_variation, SAMPLE_RATE)
    sf.write(SOUNDS_DIR / "Filtered_Bilateral.wav", bilateral, SAMPLE_RATE)
    sf.write(SOUNDS_DIR / "Filtered_Wavelet.wav", wavelet, SAMPLE_RATE)

    save_filtered_plot(time, data, invariance, "J-Invariance", "J_Invariance.png")
    save_filtered_plot(time, data, total_variation, "Total Variation", "Total_Variation.png")
    save_filtered_plot(time, data, bilateral, "Bilateral", "Bilateral.png")
    save_filtered_plot(time, data, wavelet, "Wavelet", "Wavelet.png")


def wavelet_shifted_filter():
    ensure_directories()

    if not NAME_ORIGINAL_WAV.exists():
        raise FileNotFoundError(
            f"Input file was not found: {NAME_ORIGINAL_WAV}. "
            "Copy the Sounds folder from the previous practical work first."
        )

    if not NAME_ORIGINAL_RAW.exists():
        data_for_raw, _ = sf.read(NAME_ORIGINAL_WAV)
        data_for_raw = to_mono(data_for_raw)
        data_for_raw = np.int16(np.clip(data_for_raw, -1.0, 1.0) * 32767)
        with open(NAME_ORIGINAL_RAW, "wb") as f:
            f.write(data_for_raw.tobytes())

    process_recording()
    sound_filter()

    data, fs_original = sf.read(NAME_ORIGINAL_WAV)
    data = to_mono(data)
    time = np.arange(len(data)) / fs_original

    max_shifts = [0, 1, 3, 5]
    signals = []
    for n, s in enumerate(max_shifts):
        sig_filtered = cycle_spin(
            data,
            func=wavelet_denoiser,
            max_shifts=s,
            shift_steps=5,
        )
        sf.write(SOUNDS_DIR / f"Filtered_Shifted_Wavelet_{n}.wav", sig_filtered, SAMPLE_RATE)
        signals.append(sig_filtered)

    kernel = gaussian_kernel(size=11, sigma=2)
    filtered_signal = convolve(data, kernel, mode="same")
    sf.write(SOUNDS_DIR / "Filtered_Gaussian_Filter.wav", filtered_signal, SAMPLE_RATE)

    plt.figure(figsize=(12, 6))
    plt.plot(time, data, label="Original")
    plt.plot(time, signals[0], label="Wavelet Shifted: no shift")
    plt.plot(time, signals[1], label="Wavelet Shifted: 1x2")
    plt.plot(time, signals[2], label="Wavelet Shifted: 1x4")
    plt.plot(time, signals[3], label="Wavelet Shifted: 1x6")
    plt.title("Wavelet Shift-Invariant filtering")
    plt.xlabel("Time, s")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "Shift_Invariant_Wavelet.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(time, data, label="Original")
    plt.plot(time, filtered_signal, label="Gaussian Filter")
    plt.title("Gaussian filtering")
    plt.xlabel("Time, s")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "Gaussian_Filter.png", dpi=150)
    plt.close()


def to_scientific_pretty(x, precision=2):
    if x == 0:
        return "0"

    superscripts = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    mantissa, exponent = f"{x:.{precision}e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent = str(int(exponent)).translate(superscripts)
    return f"{mantissa} * 10{exponent}"


def get_filter_name(path):
    path = Path(path)

    if path.resolve() == NAME_RESAMPLED_WAV.resolve():
        return "Ресемпл 4 кГц"

    type_filter = path.stem.replace("Filtered_", "")
    type_filter = type_filter.replace("_", " ")

    if type_filter == "4000[Hz] 2[byte]":
        return "Лінійний фільтр 4 кГц"

    return type_filter


def calculate_signal_metrics():
    ensure_directories()

    data_original, _ = sf.read(NAME_ORIGINAL_WAV)
    data_original = to_mono(data_original)

    headers = ["MSE", "MAE", "RMSE", "R2", "D"]
    rows = []
    results = []

    wav_files = sorted(glob.glob(str(SOUNDS_DIR / "*.wav")))
    for sound_path in wav_files:
        sound_path = Path(sound_path)

        if sound_path.resolve() == NAME_ORIGINAL_WAV.resolve():
            continue

        data, _ = sf.read(sound_path)
        data = to_mono(data)

        if sound_path.resolve() == NAME_RESAMPLED_WAV.resolve() or len(data) != len(data_original):
            data = resample(data, len(data_original))

        mse = mean_squared_error(data_original, data)
        mae = mean_absolute_error(data_original, data)
        rmse = np.sqrt(mse)
        r2 = r2_score(data_original, data)
        variance = np.var(data_original - data)

        rows.append(get_filter_name(sound_path))
        results.append(
            [
                to_scientific_pretty(mse),
                to_scientific_pretty(mae),
                to_scientific_pretty(rmse),
                round(r2, 2),
                to_scientific_pretty(variance),
            ]
        )

    n_rows = len(rows)
    n_cols = len(headers)
    fig, ax = plt.subplots(figsize=(n_cols * 2.8, max(n_rows * 0.4, 2)))
    ax.axis("off")

    table = ax.table(
        cellText=results,
        rowLabels=rows,
        colLabels=headers,
        cellLoc="center",
        rowLoc="center",
        bbox=[0.08, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    plt.tight_layout()
    plt.savefig(SOUNDS_DIR / "Quality_Metrics_Table.png", dpi=600, bbox_inches="tight")
    plt.close()


def calculate_metrics(original, filtered):
    mse = mean_squared_error(original, filtered)
    mae = mean_absolute_error(original, filtered)
    rmse = np.sqrt(mse)
    r2 = r2_score(original, filtered)
    variance = np.var(original - filtered)
    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "D": variance,
    }


def plot_snr_metric(metric_name, snr_values, wt_mean, wt_list, g_mean, g_list):
    snr_scatter_wt = []
    metric_scatter_wt = []
    snr_scatter_g = []
    metric_scatter_g = []

    for snr, metric_list in zip(snr_values, wt_list):
        snr_scatter_wt.extend([snr] * len(metric_list))
        metric_scatter_wt.extend(metric_list)

    for snr, metric_list in zip(snr_values, g_list):
        snr_scatter_g.extend([snr] * len(metric_list))
        metric_scatter_g.extend(metric_list)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(
        snr_scatter_wt,
        metric_scatter_wt,
        color="red",
        alpha=0.05,
        label=f"Окремі значення {metric_name} WT",
    )
    axes[0].plot(snr_values, wt_mean, linewidth=2, label=f"Середнє {metric_name} WT")
    axes[0].scatter(
        snr_scatter_g,
        metric_scatter_g,
        color="green",
        alpha=0.05,
        label=f"Окремі значення {metric_name} GF",
    )
    axes[0].plot(snr_values, g_mean, linewidth=2, label=f"Середнє {metric_name} GF")
    axes[0].set_xticks(np.arange(-10, 21, 2))
    axes[0].set_xlabel("SNR (дБ)")
    axes[0].set_ylabel(metric_name)
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_title("Лінійний масштаб")

    axes[1].scatter(
        snr_scatter_wt,
        metric_scatter_wt,
        color="red",
        alpha=0.05,
        label=f"Окремі значення {metric_name} WT",
    )
    axes[1].plot(snr_values, wt_mean, linewidth=2, label=f"Середнє {metric_name} WT")
    axes[1].scatter(
        snr_scatter_g,
        metric_scatter_g,
        color="green",
        alpha=0.05,
        label=f"Окремі значення {metric_name} GF",
    )
    axes[1].plot(snr_values, g_mean, linewidth=2, label=f"Середнє {metric_name} GF")
    axes[1].set_xticks(np.arange(-10, 21, 1))
    axes[1].set_xlabel("SNR (дБ)")
    axes[1].set_ylabel(metric_name)
    axes[1].grid(True)
    if metric_name != "R2":
        axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].set_title("Логарифмічний масштаб")

    plt.tight_layout()
    plt.savefig(SOUNDS_DIR / f"SNR_{metric_name}.png", dpi=600)
    plt.close(fig)


def filtration_efficiency():
    ensure_directories()

    data, _ = sf.read(NAME_ORIGINAL_WAV)
    data = to_mono(data)
    signal_power = np.mean(data ** 2)
    max_shifts = 5
    kernel = gaussian_kernel(size=11, sigma=2)

    metric_names = ["MSE", "MAE", "RMSE", "R2", "D"]
    wt_mean = {name: [] for name in metric_names}
    wt_list = {name: [] for name in metric_names}
    g_mean = {name: [] for name in metric_names}
    g_list = {name: [] for name in metric_names}
    snr_values = []

    for snr_db in np.arange(-10, 21, 0.5):
        wt_current = {name: [] for name in metric_names}
        g_current = {name: [] for name in metric_names}
        noise_power = signal_power / (10 ** (snr_db / 10))

        for _ in range(10):
            noise = np.random.normal(0, np.sqrt(noise_power), size=data.shape)
            noisy_signal = data + noise

            sig_filtered_wavelet = cycle_spin(
                noisy_signal,
                func=wavelet_denoiser,
                max_shifts=max_shifts,
                shift_steps=5,
                num_workers=1,
            )
            sig_filtered_gaussian = convolve(noisy_signal, kernel, mode="same")

            wt_metrics = calculate_metrics(data, sig_filtered_wavelet)
            g_metrics = calculate_metrics(data, sig_filtered_gaussian)

            for name in metric_names:
                wt_current[name].append(wt_metrics[name])
                g_current[name].append(g_metrics[name])

        for name in metric_names:
            wt_mean[name].append(np.mean(wt_current[name]))
            wt_list[name].append(list(wt_current[name]))
            g_mean[name].append(np.mean(g_current[name]))
            g_list[name].append(list(g_current[name]))

        snr_values.append(snr_db)

    for name in metric_names:
        plot_snr_metric(
            name,
            snr_values,
            wt_mean[name],
            wt_list[name],
            g_mean[name],
            g_list[name],
        )


if __name__ == "__main__":
    # wavelet_shifted_filter()
    # recognizer = srec.Recognizer()
    # microphone = srec.Microphone(device_index=MICROPHONE_INDEX, sample_rate=SAMPLE_RATE)
    # sound_recoder(recognizer, microphone)
    # calculate_signal_metrics()
    filtration_efficiency()
