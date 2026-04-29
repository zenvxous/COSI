"""Финальные графики и сводная таблица для отчёта.

Берёт результаты из results/metrics_*.csv и строит:
  • Кривые «метрика vs целевой SNR» по каждой метрике (до vs после DFN2)
  • Графики формы волны и спектрограмм для одного примера (SNR=8 дБ, white)
  • Финальную таблицу в формате методички (markdown)
"""

from __future__ import annotations

import os
import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(ROOT, "plots", "final")
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_metric_curves() -> None:
    """Сравнительные кривые метрик от SNR (две группы шума)."""
    noisy = pd.read_csv(os.path.join(RESULTS_DIR, "metrics_noisy.csv"))
    den = pd.read_csv(os.path.join(RESULTS_DIR, "metrics_denoised.csv"))

    metrics = [
        ("SNR_self", "SNR (своя реализация), дБ"),
        ("SDR_self", "SDR (своя реализация), дБ"),
        ("SI_SDR", "SI-SDR, дБ"),
        ("PESQ_wb", "PESQ (wb)"),
        ("STOI", "STOI"),
        ("NISQA", "NISQA, MOS"),
        ("DNSMOS_p808", "DNSMOS P.808 MOS"),
        ("DNSMOS_OVRL", "DNSMOS OVRL"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    for ax, (col, title) in zip(axes.flat, metrics):
        for kind, marker in [("white", "o"), ("pink", "s")]:
            n = noisy[noisy["noise_kind"] == kind].sort_values("snr_target")
            d = den[den["noise_kind"] == kind].sort_values("snr_target")
            ax.plot(n["snr_target"], n[col], marker=marker, linestyle="--",
                    label=f"{kind}, до DFN2", alpha=0.85)
            ax.plot(d["snr_target"], d[col], marker=marker, linestyle="-",
                    label=f"{kind}, после DFN2", alpha=0.95)
        ax.set_xlabel("Целевой SNR, дБ")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Объективные метрики качества: до vs после DeepFilterNet2",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "metrics_vs_snr.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {os.path.relpath(out, ROOT)}")


def plot_waveforms_example(snr: int = 8, kind: str = "white") -> None:
    """Сравнение формы волны и спектра: clean / noisy / denoised."""
    clean_path = os.path.join(ROOT, "audio", "voice", "voice_clean.wav")
    noisy_path = os.path.join(ROOT, "audio", "noisy",
                              f"voice_{kind}_snr{snr:02d}.wav")
    den_path = os.path.join(ROOT, "audio", "denoised",
                            f"voice_{kind}_snr{snr:02d}_dfn2.wav")

    sr = 16000
    clean, _ = librosa.load(clean_path, sr=sr, mono=True)
    noisy, _ = librosa.load(noisy_path, sr=sr, mono=True)
    den, _ = librosa.load(den_path, sr=sr, mono=True)
    n = min(len(clean), len(noisy), len(den))
    clean, noisy, den = clean[:n], noisy[:n], den[:n]
    t = np.arange(n) / sr

    fig, axes = plt.subplots(3, 2, figsize=(14, 9))

    titles = ["Чистый голос", f"Зашумлённый ({kind}, SNR={snr} дБ)",
              "После DeepFilterNet2"]
    for i, (sig, title) in enumerate(zip([clean, noisy, den], titles)):
        axes[i, 0].plot(t, sig, color=f"C{i}")
        axes[i, 0].set_title(title)
        axes[i, 0].set_xlabel("Время, с")
        axes[i, 0].set_ylim(-1, 1)
        axes[i, 0].grid(alpha=0.3)

        D = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=1024,
                                                         hop_length=256)),
                                     ref=np.max)
        img = librosa.display.specshow(D, sr=sr, hop_length=256,
                                        x_axis="time", y_axis="log",
                                        ax=axes[i, 1], cmap="magma")
        axes[i, 1].set_title(f"Спектрограмма — {title}")
        fig.colorbar(img, ax=axes[i, 1], format="%+2.0f dB")

    fig.suptitle(f"Сравнение волновых форм и спектров (пример: {kind}, SNR={snr} дБ)",
                 fontsize=13)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"waveform_compare_{kind}_snr{snr:02d}.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {os.path.relpath(out, ROOT)}")


def make_markdown_table() -> None:
    """Сохраняет markdown-таблицу из методички."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "metrics_combined.csv"))

    def make_table(prefix: str, title: str) -> str:
        # Колонки в формате методички
        cols_show = [
            ("snr_target",       "Тарг. SNR"),
            (f"SNR_self_{prefix}",   "SNR"),
            (f"SDR_self_{prefix}",   "SDR"),
            (f"SI_SDR_{prefix}",     "SI-SDR"),
            (f"PESQ_wb_{prefix}",    "PESQ"),
            (f"NISQA_{prefix}",      "NISQA"),
            (f"DNSMOS_OVRL_{prefix}","DNSMOS_OVRL"),
        ]
        head = ("| Файл | " + " | ".join(c[1] for c in cols_show) + " |\n"
                + "| --- |" + " --- |" * len(cols_show) + "\n")
        rows = []
        for _, r in df.iterrows():
            row = [r["file"]] + [f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])
                                  for c, _ in cols_show]
            rows.append("| " + " | ".join(row) + " |")
        return f"### {title}\n\n" + head + "\n".join(rows) + "\n"

    md = "# Таблица метрик качества звука\n\n"
    md += make_table("noisy", "До шумоподавления (зашумлённые файлы)")
    md += "\n"
    md += make_table("dfn2", "После DeepFilterNet2 (очищенные файлы)")

    out = os.path.join(RESULTS_DIR, "metrics_table.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  saved {os.path.relpath(out, ROOT)}")


def make_summary_csv() -> None:
    """Сводная статистика: средний прирост по каждой метрике."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "metrics_combined.csv"))
    cols = ["SNR_self", "SDR_self", "SI_SDR", "PESQ_wb", "STOI", "NISQA",
            "DNSMOS_p808", "DNSMOS_SIG", "DNSMOS_BAK", "DNSMOS_OVRL"]
    rows = []
    for c in cols:
        diff = df[f"{c}_dfn2"] - df[f"{c}_noisy"]
        rows.append({
            "metric": c,
            "mean_noisy": df[f"{c}_noisy"].mean(),
            "mean_denoised": df[f"{c}_dfn2"].mean(),
            "mean_improvement": diff.mean(),
            "improvement_white": diff[df["noise_kind"] == "white"].mean(),
            "improvement_pink": diff[df["noise_kind"] == "pink"].mean(),
        })
    out = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    out.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  saved {os.path.relpath(out_path, ROOT)}")
    print(out.to_string(index=False))


def main() -> None:
    print("=== Финальные графики и таблицы ===")
    plot_metric_curves()
    for kind in ("white", "pink"):
        for snr in (4, 12, 22):
            plot_waveforms_example(snr=snr, kind=kind)
    make_markdown_table()
    make_summary_csv()


if __name__ == "__main__":
    main()
