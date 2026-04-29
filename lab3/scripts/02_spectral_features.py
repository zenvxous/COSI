"""Пункт 3 задания: спектральные признаки аудиосигнала.

Самостоятельно реализованы (по варианту 4):
  • Спектральная ширина (Spectral Bandwidth)
  • Частота пересечения нуля (Zero-Crossing Rate)

Через librosa:
  • Спектральный центроид
  • Спектральный спад (Spectral Roll-off)
  • Мел-частотные кепстральные коэффициенты (MFCC)
  • Цветность (Chroma)

Для каждого музыкального файла строим один общий рисунок с графиками
всех признаков, плюс отдельную сводную таблицу средних значений
для качественного сравнения жанров.
"""

from __future__ import annotations

import os
import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsp_utils import spectral_bandwidth, zero_crossing_rate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUSIC_DIR = os.path.join(ROOT, "audio", "music")
PLOTS_DIR = os.path.join(ROOT, "plots", "features")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

N_FFT = 2048
HOP = 512


def compute_features(y: np.ndarray, sr: int) -> dict:
    """Считает все 6 признаков для одного сигнала."""
    # ===== Самостоятельно =====
    bw_my = spectral_bandwidth(y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    zcr_my = zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)

    # ===== librosa =====
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, roll_percent=0.85)[0]
    bw_lib = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
    zcr_lib = librosa.feature.zero_crossing_rate(
        y=y, frame_length=N_FFT, hop_length=HOP)[0]
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)

    return {
        "centroid": centroid, "rolloff": rolloff,
        "bw_my": bw_my, "bw_lib": bw_lib,
        "zcr_my": zcr_my, "zcr_lib": zcr_lib,
        "mfcc": mfcc, "chroma": chroma,
    }


def plot_features(name: str, sr: int, feats: dict, duration: float) -> str:
    """Строит сетку графиков всех признаков."""
    times = np.linspace(0, duration, len(feats["centroid"]))
    times_zcr = np.linspace(0, duration, len(feats["zcr_my"]))

    fig = plt.figure(figsize=(15, 11))

    # 1. Centroid
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(times, feats["centroid"], color="C0")
    ax1.set_title("Спектральный центроид (librosa), Гц")
    ax1.set_xlabel("Время, с"); ax1.grid(alpha=0.3)

    # 2. Rolloff
    ax2 = fig.add_subplot(4, 2, 2)
    ax2.plot(times, feats["rolloff"], color="C1")
    ax2.set_title("Спектральный спад 85% (librosa), Гц")
    ax2.set_xlabel("Время, с"); ax2.grid(alpha=0.3)

    # 3. Bandwidth — своя реализация vs librosa
    ax3 = fig.add_subplot(4, 2, 3)
    ax3.plot(times, feats["bw_my"], label="своя", color="C2")
    ax3.plot(times, feats["bw_lib"], label="librosa", color="C3",
             linestyle="--", alpha=0.7)
    ax3.set_title("Спектральная ширина, Гц (своя vs librosa)")
    ax3.set_xlabel("Время, с"); ax3.legend(); ax3.grid(alpha=0.3)

    # 4. ZCR — своя реализация vs librosa
    ax4 = fig.add_subplot(4, 2, 4)
    ax4.plot(times_zcr, feats["zcr_my"], label="своя", color="C4")
    ax4.plot(times_zcr, feats["zcr_lib"], label="librosa", color="C5",
             linestyle="--", alpha=0.7)
    ax4.set_title("Частота пересечения нуля (своя vs librosa)")
    ax4.set_xlabel("Время, с"); ax4.legend(); ax4.grid(alpha=0.3)

    # 5. MFCC
    ax5 = fig.add_subplot(4, 2, (5, 6))
    img5 = librosa.display.specshow(feats["mfcc"], sr=sr, hop_length=HOP,
                                    x_axis="time", ax=ax5, cmap="viridis")
    ax5.set_title("MFCC (13 коэффициентов)")
    ax5.set_ylabel("№ коэффициента")
    fig.colorbar(img5, ax=ax5)

    # 6. Chroma
    ax6 = fig.add_subplot(4, 2, (7, 8))
    img6 = librosa.display.specshow(feats["chroma"], sr=sr, hop_length=HOP,
                                    x_axis="time", y_axis="chroma",
                                    ax=ax6, cmap="magma")
    ax6.set_title("Цветность (chroma_stft)")
    fig.colorbar(img6, ax=ax6)

    fig.suptitle(f"Спектральные признаки — {name}", fontsize=14, y=1.00)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{name}.png")
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    print("=== Пункт 3 задания: спектральные признаки ===")
    summary_rows = []

    music_files = sorted(
        os.path.join(MUSIC_DIR, f)
        for f in os.listdir(MUSIC_DIR) if f.endswith(".wav")
    )
    for wav in music_files:
        name = os.path.splitext(os.path.basename(wav))[0]
        y, sr = librosa.load(wav, sr=None, mono=True)
        feats = compute_features(y, sr)
        out = plot_features(name, sr, feats, duration=len(y) / sr)
        print(f"  [+] {name} -> {os.path.relpath(out, ROOT)}")

        # Совпадение «своя vs librosa» для отчёта
        bw_diff = float(np.mean(np.abs(feats["bw_my"] - feats["bw_lib"])))
        zcr_diff = float(np.mean(np.abs(feats["zcr_my"] - feats["zcr_lib"])))

        summary_rows.append({
            "file": name,
            "centroid_mean_Hz": float(feats["centroid"].mean()),
            "rolloff85_mean_Hz": float(feats["rolloff"].mean()),
            "bandwidth_mean_Hz": float(feats["bw_my"].mean()),
            "bandwidth_diff_vs_librosa": bw_diff,
            "zcr_mean": float(feats["zcr_my"].mean()),
            "zcr_diff_vs_librosa": zcr_diff,
            "mfcc1_mean": float(feats["mfcc"][0].mean()),
            "chroma_dominant_pc": int(np.argmax(feats["chroma"].mean(axis=1))),
        })

    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(RESULTS_DIR, "spectral_features_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nСводка средних: {os.path.relpath(csv_path, ROOT)}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
