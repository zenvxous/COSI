"""Пункт 6 задания: шумоподавление через DeepFilterNet2.

Загружаем модель DeepFilterNet2, обрабатываем все зашумлённые файлы из
audio/noisy/, сохраняем очищенные версии в audio/denoised/.
DeepFilterNet работает на 48 кГц, поэтому ресэмплируем туда и обратно.
"""

from __future__ import annotations

import os
import sys
import warnings
import logging

import numpy as np
import soundfile as sf
import librosa
import torch

warnings.filterwarnings("ignore")
logging.getLogger("DF").setLevel(logging.ERROR)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOISY_DIR = os.path.join(ROOT, "audio", "noisy")
DENOISED_DIR = os.path.join(ROOT, "audio", "denoised")
os.makedirs(DENOISED_DIR, exist_ok=True)


def main() -> None:
    print("=== Пункт 6 задания: DeepFilterNet2 ===")

    from df.enhance import enhance, init_df

    # Используем именно DeepFilterNet2 как в задании
    model, df_state, _ = init_df(model_base_dir="DeepFilterNet2")
    df_sr = df_state.sr()
    print(f"  Модель загружена. Внутренняя частота: {df_sr} Гц")

    files = sorted(f for f in os.listdir(NOISY_DIR) if f.endswith(".wav"))
    for fname in files:
        in_path = os.path.join(NOISY_DIR, fname)
        out_path = os.path.join(DENOISED_DIR, fname.replace(".wav", "_dfn2.wav"))

        # Загружаем noisy и приводим к 48 кГц mono float32
        y, sr = librosa.load(in_path, sr=df_sr, mono=True)
        audio = torch.from_numpy(y).unsqueeze(0)

        with torch.no_grad():
            enhanced = enhance(model, df_state, audio)

        enhanced_np = enhanced.squeeze(0).numpy()

        # Сохраняем обратно на исходной частоте 16 кГц для совместимости с
        # PESQ/STOI и метриками в исходной шкале
        enhanced_16k = librosa.resample(enhanced_np, orig_sr=df_sr, target_sr=16000)
        sf.write(out_path, enhanced_16k.astype(np.float32), 16000,
                 subtype="PCM_16")
        print(f"  [+] {fname}  ->  {os.path.basename(out_path)}")

    print(f"\nГотово. Очищенные файлы: {DENOISED_DIR}")


if __name__ == "__main__":
    main()
