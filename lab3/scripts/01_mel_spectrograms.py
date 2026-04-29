"""Пункт 2 задания: построение мел-спектрограмм для 4 музыкальных жанров.

Для каждого файла строим:
  • Самостоятельную реализацию мел-спектрограммы (`dsp_utils.melspectrogram`)
  • Реализацию через librosa (`librosa.feature.melspectrogram`)
  • Сравнительный график

Дополнительно сохраняем график банка мел-фильтров (рисунки 22 и 23 из методички).
"""

from __future__ import annotations

import os
import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsp_utils import (
    melspectrogram,
    mel_filterbank,
    power_to_db,
    hz_to_mel,
    mel_to_hz,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUSIC_DIR = os.path.join(ROOT, "audio", "music")
PLOTS_DIR = os.path.join(ROOT, "plots", "mel")
os.makedirs(PLOTS_DIR, exist_ok=True)

N_FFT = 2048
HOP = 512
N_MELS = 128


def plot_filterbank(sr: int = 22050) -> None:
    """График банка мел-фильтров на оси мел и на оси Гц (рисунки 22, 23)."""
    fb = mel_filterbank(n_mels=10, n_fft=N_FFT, sr=sr, norm_slaney=False)
    fft_freqs = np.linspace(0, sr / 2, fb.shape[1])
    mel_freqs = hz_to_mel(fft_freqs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for m in range(fb.shape[0]):
        axes[0].plot(mel_freqs, fb[m])
        axes[1].plot(fft_freqs, fb[m])
    axes[0].set_xlabel("Частота (Mel)")
    axes[0].set_ylabel("Амплитуда фильтра (вес)")
    axes[0].set_title("Банк из 10 мел-фильтров (мел-шкала)")
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Частота (Hz)")
    axes[1].set_ylabel("Амплитуда фильтра (вес)")
    axes[1].set_title("Тот же банк после преобразования в герцы")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "00_mel_filterbank.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  saved {os.path.relpath(out, ROOT)}")


def process_file(wav_path: str) -> None:
    name = os.path.splitext(os.path.basename(wav_path))[0]
    print(f"\n[файл] {name}")

    y, sr = librosa.load(wav_path, sr=None, mono=True)

    # 1. Самостоятельная мел-спектрограмма
    M_my = melspectrogram(y, sr=sr, n_fft=N_FFT, hop_length=HOP,
                          n_mels=N_MELS)
    M_my_db = power_to_db(M_my, ref="max")

    # 2. Через librosa
    M_lib = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                           hop_length=HOP, n_mels=N_MELS,
                                           power=2.0)
    M_lib_db = librosa.power_to_db(M_lib, ref=np.max)

    # 3. Графики
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2))
    img1 = librosa.display.specshow(M_my_db, sr=sr, hop_length=HOP,
                                    x_axis="time", y_axis="mel",
                                    fmax=sr / 2, ax=axes[0], cmap="magma")
    axes[0].set_title(f"{name} — своя реализация")
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")

    img2 = librosa.display.specshow(M_lib_db, sr=sr, hop_length=HOP,
                                    x_axis="time", y_axis="mel",
                                    fmax=sr / 2, ax=axes[1], cmap="magma")
    axes[1].set_title(f"{name} — librosa")
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")

    diff = M_my_db - M_lib_db
    img3 = axes[2].imshow(diff, origin="lower", aspect="auto",
                          cmap="seismic", vmin=-10, vmax=10)
    axes[2].set_title("Разница (своя − librosa), дБ")
    axes[2].set_xlabel("Кадры")
    axes[2].set_ylabel("Мел-индекс")
    fig.colorbar(img3, ax=axes[2])

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{name}.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  saved {os.path.relpath(out, ROOT)}")
    print(f"  shapes: my={M_my.shape}, lib={M_lib.shape}, "
          f"mean |dB diff|={np.mean(np.abs(diff)):.3f}")


def main() -> None:
    print("=== Пункт 2 задания: мел-спектрограммы ===")
    plot_filterbank(sr=22050)

    music_files = sorted(
        os.path.join(MUSIC_DIR, f)
        for f in os.listdir(MUSIC_DIR) if f.endswith(".wav")
    )
    for wav in music_files:
        process_file(wav)

    print("\nГотово. Графики -> plots/mel/")


if __name__ == "__main__":
    main()
