"""Пункт 4 задания: смешивание чистого голоса и шума по заданному SNR.

Делаем полный набор зашумлённых файлов в диапазоне SNR от 4 до 22 дБ с шагом 2 дБ.
Для каждого уровня SNR смешиваем голос с двумя видами шума:
  * белый
  * розовый (1/f)

Сохраняем wav-файлы и проверяем фактический SNR (он должен совпадать с целевым).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsp_utils import mix_with_snr, generate_noise, snr_db

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_PATH = os.path.join(ROOT, "audio", "voice", "voice_clean.wav")
NOISY_DIR = os.path.join(ROOT, "audio", "noisy")
PLOTS_DIR = os.path.join(ROOT, "plots", "noise")
os.makedirs(NOISY_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

SNR_RANGE = list(range(4, 23, 2))  # 4, 6, 8, ..., 22  → 10 уровней
NOISE_KINDS = ["white", "pink"]


def main() -> None:
    print("=== Пункт 4 задания: смешивание голоса с шумом ===")
    voice, sr = librosa.load(VOICE_PATH, sr=None, mono=True)
    voice = voice.astype(np.float32)
    print(f"  голос: sr={sr}, длительность={len(voice)/sr:.2f} с")

    rng = np.random.default_rng(42)

    rows = []
    for kind in NOISE_KINDS:
        # Один шум на все SNR уровни (для удобства сравнения)
        noise = generate_noise(len(voice), kind=kind, rng=rng)
        for snr in SNR_RANGE:
            noisy, scaled_noise = mix_with_snr(voice, noise, snr_db=snr, rng=rng)
            measured = snr_db(voice, noisy)
            out_name = f"voice_{kind}_snr{snr:02d}.wav"
            out_path = os.path.join(NOISY_DIR, out_name)
            sf.write(out_path, noisy, sr, subtype="PCM_16")
            rows.append((kind, snr, measured))
            print(f"  [+] {out_name:30s} target={snr:3d} dB  measured={measured:6.2f} dB")

    # Графики
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    examples_to_plot = [
        ("white", 4),
        ("white", 22),
        ("pink", 4),
        ("pink", 22),
    ]
    for ax, (kind, snr) in zip(axes.flat, examples_to_plot):
        wav_path = os.path.join(NOISY_DIR, f"voice_{kind}_snr{snr:02d}.wav")
        y, sr = sf.read(wav_path)
        t = np.arange(len(y)) / sr
        ax.plot(t, voice, color="C0", alpha=0.7, label="чистый голос")
        ax.plot(t, y - voice, color="C3", alpha=0.6, label="шум")
        ax.set_title(f"{kind} шум, SNR = {snr} дБ")
        ax.set_xlabel("Время, с")
        ax.set_ylabel("Амплитуда")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Чистый голос vs выделенный шум (для разных уровней SNR)",
                 fontsize=13)
    plt.tight_layout()
    out_plot = os.path.join(PLOTS_DIR, "snr_examples_waveform.png")
    plt.savefig(out_plot, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  График: {os.path.relpath(out_plot, ROOT)}")
    print(f"  Всего создано {len(rows)} зашумлённых файлов в {NOISY_DIR}")


if __name__ == "__main__":
    main()
