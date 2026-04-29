"""Пункт 5 задания: сравнение методов оценки качества звука.

Считает все объективные метрики на двух наборах:
  • noisy/        — зашумлённые файлы (до шумоподавления)
  • denoised/     — обработанные DeepFilterNet2

Метрики:
  • SNR        — реализована самостоятельно
  • SDR        — реализована самостоятельно
  • SI-SDR     — через библиотеку (для справки, методичка тоже его требует)
  • PESQ (wb)  — через pesq
  • STOI       — через pystoi (бонус)
  • NISQA      — нейросетевая неинтрузивная метрика
  • DNSMOS     — нейросетевая, P.835 (SIG / BAK / OVRL) + P.808

Результат — две CSV-таблицы (до/после) и сводный график зависимости метрик
от исходного SNR.
"""

from __future__ import annotations

import os
import sys
import warnings
import logging

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsp_utils import snr_db, sdr_db

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_PATH = os.path.join(ROOT, "audio", "voice", "voice_clean.wav")
NOISY_DIR = os.path.join(ROOT, "audio", "noisy")
DENOISED_DIR = os.path.join(ROOT, "audio", "denoised")
RESULTS_DIR = os.path.join(ROOT, "results")
NISQA_WEIGHTS = os.path.join(ROOT, "models", "nisqa", "nisqa.tar")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# SI-SDR (масштабно-инвариантный SDR) — самостоятельно как бонус
# ------------------------------------------------------------------

def si_sdr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """SI-SDR (Le Roux et al., 2019).

    Не зависит от общего масштаба estimate, поэтому хорошо подходит
    для оценки speech enhancement, где модель может слегка изменить
    громкость сигнала.
    """
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    ref -= ref.mean()
    est -= est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    s_target = alpha * ref
    e_noise = est - s_target
    return float(10.0 * np.log10(
        (np.sum(s_target ** 2) + 1e-12) / (np.sum(e_noise ** 2) + 1e-12)
    ))


# ------------------------------------------------------------------
# DNSMOS — через torchmetrics
# ------------------------------------------------------------------

def dnsmos_scores(y: np.ndarray, sr: int) -> tuple[float, float, float, float]:
    """Возвращает (P.808 MOS, MOS_SIG, MOS_BAK, MOS_OVRL)."""
    from torchmetrics.functional.audio.dnsmos import (
        deep_noise_suppression_mean_opinion_score,
    )
    if sr != 16000:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)
    out = deep_noise_suppression_mean_opinion_score(
        torch.from_numpy(y).unsqueeze(0).float(),
        fs=16000, personalized=False,
    )[0].cpu().numpy()
    return float(out[0]), float(out[1]), float(out[2]), float(out[3])


# ------------------------------------------------------------------
# NISQA — общий класс предиктора, чтобы не загружать модель на каждый файл
# ------------------------------------------------------------------

class NISQAPredictor:
    """Обёртка вокруг NISQA для предсказания одной MOS на файл."""

    def __init__(self, weights_path: str):
        from nisqa.NISQA_model import nisqaModel
        self._cls = nisqaModel
        self._weights = weights_path
        self._model = None

    def _ensure(self, file_path: str):
        # NISQA при инициализации сама загружает датасет, поэтому проще
        # каждый раз создавать новый экземпляр (стоимость ~0.5 с — приемлемо).
        args = {
            "mode": "predict_file",
            "pretrained_model": self._weights,
            "deg": file_path,
            "data_dir": None,
            "output_dir": None,
            "tr_bs_val": 1,
            "tr_num_workers": 0,
            "ms_channel": None,
        }
        return self._cls(args)

    def predict(self, file_path: str) -> float:
        m = self._ensure(file_path)
        df = m.predict()
        return float(df["mos_pred"].iloc[0])


# ------------------------------------------------------------------
# Основной обход файлов
# ------------------------------------------------------------------

METRIC_COLS = [
    "noise_kind", "snr_target",
    "SNR_self", "SDR_self", "SI_SDR",
    "PESQ_wb", "STOI",
    "NISQA",
    "DNSMOS_p808", "DNSMOS_SIG", "DNSMOS_BAK", "DNSMOS_OVRL",
]


def parse_name(fname: str) -> tuple[str, int] | None:
    """Из 'voice_white_snr10.wav' / '..._dfn2.wav' получаем ('white', 10)."""
    base = fname.replace(".wav", "").replace("_dfn2", "")
    parts = base.split("_")
    if len(parts) < 3 or not parts[2].startswith("snr"):
        return None
    return parts[1], int(parts[2][3:])


def compute_row(degraded_path: str, clean: np.ndarray, sr: int,
                nisqa_pred: NISQAPredictor) -> dict:
    deg, deg_sr = sf.read(degraded_path)
    if deg_sr != sr:
        deg = librosa.resample(deg.astype(np.float32),
                               orig_sr=deg_sr, target_sr=sr)
    n = min(len(clean), len(deg))
    ref, est = clean[:n], deg[:n]

    row = {
        "SNR_self":  snr_db(ref, est),
        "SDR_self":  sdr_db(ref, est),
        "SI_SDR":    si_sdr_db(ref, est),
        "PESQ_wb":   float("nan"),
        "STOI":      float("nan"),
        "NISQA":     float("nan"),
        "DNSMOS_p808": float("nan"),
        "DNSMOS_SIG":  float("nan"),
        "DNSMOS_BAK":  float("nan"),
        "DNSMOS_OVRL": float("nan"),
    }
    try:
        row["PESQ_wb"] = float(pesq(sr, ref, est, "wb"))
    except Exception as e:
        print(f"   PESQ error: {e}")
    try:
        row["STOI"] = float(stoi(ref, est, sr, extended=False))
    except Exception as e:
        print(f"   STOI error: {e}")
    try:
        p808, sig, bak, ovrl = dnsmos_scores(est, sr)
        row["DNSMOS_p808"] = p808
        row["DNSMOS_SIG"] = sig
        row["DNSMOS_BAK"] = bak
        row["DNSMOS_OVRL"] = ovrl
    except Exception as e:
        print(f"   DNSMOS error: {e}")
    try:
        row["NISQA"] = nisqa_pred.predict(degraded_path)
    except Exception as e:
        print(f"   NISQA error: {e}")
    return row


def collect(directory: str, label: str, clean: np.ndarray, sr: int,
            nisqa_pred: NISQAPredictor) -> pd.DataFrame:
    print(f"\n=== Сбор метрик: {label} ===")
    rows = []
    for f in sorted(os.listdir(directory)):
        if not f.endswith(".wav"):
            continue
        info = parse_name(f)
        if info is None:
            continue
        kind, snr = info
        print(f"  [{label}] {f}")
        full = os.path.join(directory, f)
        row = compute_row(full, clean, sr, nisqa_pred)
        row["file"] = f
        row["noise_kind"] = kind
        row["snr_target"] = snr
        rows.append(row)
    df = pd.DataFrame(rows)
    cols = ["file"] + METRIC_COLS
    return df[cols]


def main() -> None:
    print("Загрузка чистого голоса...")
    clean, sr = librosa.load(VOICE_PATH, sr=None, mono=True)
    print(f"  sr={sr}, len={len(clean)/sr:.2f} s")

    nisqa_pred = NISQAPredictor(NISQA_WEIGHTS)

    # Опорная метрика на самом эталоне (sanity check)
    print("\nSanity check на чистом голосе:")
    p808, sig, bak, ovrl = dnsmos_scores(clean, sr)
    print(f"  DNSMOS clean: P808={p808:.2f} SIG={sig:.2f} "
          f"BAK={bak:.2f} OVRL={ovrl:.2f}")

    df_noisy = collect(NOISY_DIR, "noisy", clean, sr, nisqa_pred)
    df_denoised = collect(DENOISED_DIR, "denoised (DeepFilterNet2)",
                          clean, sr, nisqa_pred)

    df_noisy.to_csv(os.path.join(RESULTS_DIR, "metrics_noisy.csv"), index=False)
    df_denoised.to_csv(os.path.join(RESULTS_DIR, "metrics_denoised.csv"),
                       index=False)
    print(f"\nNoisy:    {len(df_noisy)} строк")
    print(f"Denoised: {len(df_denoised)} строк")

    # Сводный CSV: сравнение до/после
    merged = df_noisy.copy()
    merged = merged.rename(columns={c: f"{c}_noisy" for c in METRIC_COLS
                                    if c not in ("noise_kind", "snr_target")})
    den = df_denoised.rename(columns={c: f"{c}_dfn2" for c in METRIC_COLS
                                      if c not in ("noise_kind", "snr_target")})
    den["file"] = den["file"].str.replace("_dfn2", "")
    merged = merged.merge(den.drop(columns=["noise_kind", "snr_target"]),
                          on="file")
    merged.to_csv(os.path.join(RESULTS_DIR, "metrics_combined.csv"),
                  index=False)
    print(f"Сводно:   {len(merged)} строк")
    print("\nГотово. Файлы в results/")


if __name__ == "__main__":
    main()
