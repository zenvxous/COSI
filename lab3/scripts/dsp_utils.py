"""Общие утилиты ЦОС для лабораторной №3.

Здесь собраны самостоятельные реализации основных алгоритмов
(STFT, мел-спектрограмма, спектральная ширина, ZCR, SNR, SDR, смешивание шума).
Везде используется только numpy — без вызова librosa-аналогов.
"""

from __future__ import annotations

import numpy as np


# ======================================================================
#                       Оконное преобразование Фурье
# ======================================================================

def frame_signal(x: np.ndarray, frame_length: int, hop_length: int,
                 center: bool = True) -> np.ndarray:
    """Разбивает сигнал на кадры (фреймы).

    Возвращает массив формы (n_frames, frame_length).
    Если center=True — сигнал отражённо паддится так, чтобы кадр с индексом m
    был центрирован вокруг отсчёта m * hop_length (как в librosa).
    """
    if center:
        pad = frame_length // 2
        x = np.pad(x, pad, mode="reflect")

    n_frames = 1 + (len(x) - frame_length) // hop_length
    if n_frames < 1:
        return np.empty((0, frame_length), dtype=x.dtype)

    # Эффективный strided view
    shape = (n_frames, frame_length)
    strides = (x.strides[0] * hop_length, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return frames.copy()  # копия, чтобы избежать сюрпризов с памятью


def hann_window(N: int, periodic: bool = True) -> np.ndarray:
    """Окно Ханна длины N.

    periodic=True (по умолчанию) — окно эквивалентно scipy.signal.get_window('hann', N)
    и используется в librosa: w[n] = 0.5 - 0.5*cos(2*pi*n / N).
    periodic=False — классическое симметричное определение (через N-1).
    """
    if N <= 1:
        return np.ones(N)
    n = np.arange(N)
    denom = N if periodic else N - 1
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / denom)


def stft(x: np.ndarray, n_fft: int = 1024, hop_length: int = 256,
         window: str = "hann", center: bool = True) -> np.ndarray:
    """Самостоятельная реализация STFT (формула 41 из методички).

    Возвращает комплексный массив формы (1 + n_fft // 2, n_frames).
    """
    if window == "hann":
        w = hann_window(n_fft)
    else:
        raise ValueError(f"Окно {window!r} не реализовано")

    frames = frame_signal(x, frame_length=n_fft, hop_length=hop_length,
                          center=center)
    frames = frames * w[np.newaxis, :]
    # rfft даёт только неотрицательные частоты; этого достаточно для спектра.
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    return spec.T  # (n_freqs, n_frames)


# ======================================================================
#                          Мел-фильтры и мел-спектрограмма
# ======================================================================

def hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    """Формула (43) из методички: mel = 2595 * log10(1 + f/700)."""
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    """Формула (44) из методички: f = 700 * (10^(mel/2595) - 1)."""
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, sr: int,
                   fmin: float = 0.0,
                   fmax: float | None = None,
                   norm_slaney: bool = True) -> np.ndarray:
    """Банк треугольных мел-фильтров.

    Алгоритм из методички (стр. 53):
      1. Пересчитать fmin, fmax в мелы.
      2. Сформировать (n_mels + 2) равномерно расположенных точек на мел-шкале.
      3. Перевести их обратно в герцы и в индексы FFT-бинов.
      4. Построить треугольные фильтры между соседними точками.

    Возвращает массив формы (n_mels, n_fft // 2 + 1).
    Опционально применяется нормировка Slaney (как в librosa по умолчанию):
    каждый фильтр делится на собственную ширину в герцах. Это компенсирует
    рост ширины треугольников с частотой и обеспечивает примерно одинаковую
    «энергетическую» чувствительность всех мел-полос.
    """
    if fmax is None:
        fmax = sr / 2.0

    n_freqs = 1 + n_fft // 2
    fft_freqs = np.linspace(0, sr / 2.0, n_freqs)

    # n_mels + 2 точки: левая граница первого фильтра, центры, правая граница последнего
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for m in range(n_mels):
        f_left, f_center, f_right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        # Восходящий склон
        left_slope = (fft_freqs - f_left) / (f_center - f_left + 1e-12)
        # Нисходящий склон
        right_slope = (f_right - fft_freqs) / (f_right - f_center + 1e-12)
        fb[m] = np.maximum(0.0, np.minimum(left_slope, right_slope))

    if norm_slaney:
        enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels])
        fb *= enorm[:, np.newaxis]

    return fb


def melspectrogram(x: np.ndarray, sr: int, n_fft: int = 1024,
                   hop_length: int = 256, n_mels: int = 128,
                   fmin: float = 0.0, fmax: float | None = None,
                   power: float = 2.0) -> np.ndarray:
    """Самостоятельная реализация мел-спектрограммы (формулы 40–45).

    Возвращает массив формы (n_mels, n_frames) с энергиями (до логарифма).
    Чтобы получить картинку в дБ — применить power_to_db().
    """
    spec = stft(x, n_fft=n_fft, hop_length=hop_length, center=True)
    # Спектр мощности |X|^2 (формула 42)
    S = np.abs(spec) ** power
    fb = mel_filterbank(n_mels=n_mels, n_fft=n_fft, sr=sr,
                        fmin=fmin, fmax=fmax)
    mel = fb @ S  # (n_mels, n_frames)
    return mel


def power_to_db(S: np.ndarray, ref: float | str = "max",
                amin: float = 1e-10, top_db: float = 80.0) -> np.ndarray:
    """Преобразование мощности в дБ (повторяет librosa.power_to_db)."""
    S = np.asarray(S)
    magnitude = np.maximum(S, amin)
    if ref == "max":
        ref_value = magnitude.max()
    else:
        ref_value = float(ref)
    log_spec = 10.0 * np.log10(magnitude) - 10.0 * np.log10(max(ref_value, amin))
    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


# ======================================================================
#                       Спектральные признаки
# ======================================================================

def spectral_bandwidth(x: np.ndarray, sr: int, n_fft: int = 1024,
                       hop_length: int = 256, p: int = 2) -> np.ndarray:
    """Спектральная ширина (spectral bandwidth) — самостоятельная реализация.

    bw[m] = ( sum_k |X[m,k]| * (f[k] - centroid[m])^p / sum_k |X[m,k]| )^(1/p)

    Возвращает массив формы (n_frames,).
    """
    spec = np.abs(stft(x, n_fft=n_fft, hop_length=hop_length, center=True))
    # Нормализуем по кадрам (распределение энергии по частотам)
    freqs = np.linspace(0, sr / 2.0, spec.shape[0])
    eps = 1e-12
    norm = spec.sum(axis=0, keepdims=True) + eps
    prob = spec / norm
    centroid = (freqs[:, np.newaxis] * prob).sum(axis=0)
    deviation = np.abs(freqs[:, np.newaxis] - centroid[np.newaxis, :]) ** p
    bw = (prob * deviation).sum(axis=0) ** (1.0 / p)
    return bw


def zero_crossing_rate(x: np.ndarray, frame_length: int = 2048,
                       hop_length: int = 512,
                       center: bool = True) -> np.ndarray:
    """Частота пересечения нуля (Zero-Crossing Rate) — самостоятельно.

    Для каждого кадра считаем количество смен знака между соседними отсчётами
    и нормируем на длину кадра.
    """
    frames = frame_signal(x, frame_length=frame_length, hop_length=hop_length,
                          center=center)
    if frames.size == 0:
        return np.array([])
    # Знак (как в librosa): sign(0)=0, поэтому используем np.signbit, который
    # возвращает True для отрицательных. Считаем количество переходов True/False.
    signs = np.signbit(frames)
    crossings = np.diff(signs.astype(np.int8), axis=1) != 0
    zcr = crossings.sum(axis=1) / frame_length
    return zcr


# ======================================================================
#                  Смешивание сигнала с шумом по SNR
# ======================================================================

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def mix_with_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float,
                 rng: np.random.Generator | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Смешивает чистый сигнал и шум так, чтобы их SNR равнялся snr_db.

    Возвращает (noisy, scaled_noise). Шум обрезается/зацикливается до длины
    clean. Масштаб шума считается через RMS:
        SNR = 20 * log10(RMS_signal / RMS_noise)
        => RMS_noise_target = RMS_signal / 10^(SNR/20)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n = len(clean)
    if len(noise) < n:
        # Зацикливаем шум; на каждой итерации со случайного смещения
        repeats = int(np.ceil(n / len(noise)))
        noise = np.tile(noise, repeats)
    if len(noise) > n:
        # Случайный отрезок длины n
        start = rng.integers(0, len(noise) - n + 1)
        noise = noise[start:start + n]

    rms_signal = _rms(clean)
    rms_noise = _rms(noise)
    target_rms_noise = rms_signal / (10 ** (snr_db / 20.0))
    scale = target_rms_noise / (rms_noise + 1e-12)
    noise_scaled = noise * scale
    noisy = clean + noise_scaled
    return noisy.astype(np.float32), noise_scaled.astype(np.float32)


def generate_noise(n: int, kind: str = "white",
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Генерация синтетического шума.

    kind:
      'white' — гауссов белый шум.
      'pink'  — розовый шум (1/f) методом фильтрации белого шума в частотной
                области (амплитуда |H(f)| ~ 1/sqrt(f)).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    white = rng.standard_normal(n).astype(np.float32)
    if kind == "white":
        return white
    if kind == "pink":
        # Спектральное окрашивание
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n)
        freqs[0] = freqs[1]  # избегаем деления на ноль
        spec /= np.sqrt(freqs)
        pink = np.fft.irfft(spec, n=n).astype(np.float32)
        # Нормализация на единичную дисперсию
        pink /= (pink.std() + 1e-12)
        return pink
    raise ValueError(f"Неизвестный тип шума: {kind!r}")


# ======================================================================
#                     Метрики качества сигнала
# ======================================================================

def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Обычное отношение сигнал/шум во временной области.

    SNR = 10 * log10( ||reference||^2 / ||reference - estimate||^2 )
    """
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    n = min(len(reference), len(estimate))
    reference = reference[:n]
    estimate = estimate[:n]
    noise = reference - estimate
    num = np.sum(reference ** 2) + 1e-12
    den = np.sum(noise ** 2) + 1e-12
    return float(10.0 * np.log10(num / den))


def sdr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Signal-to-Distortion Ratio (классическое определение BSS-Eval).

    Раскладываем оценку: estimate = s_target + e_distortion, где
        s_target = <estimate, reference> / ||reference||^2 * reference
    т.е. ортогональная проекция estimate на ось reference.
    Отличие от SI-SDR: здесь reference используется в исходном масштабе.
    Здесь же фактически SDR ≈ SI-SDR с отдельным масштабом — для одного
    исходного источника они совпадают по формуле. Реализуем строго так:
        SDR = 10 * log10( ||s_target||^2 / ||e_distortion||^2 )
    """
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    n = min(len(reference), len(estimate))
    reference = reference[:n]
    estimate = estimate[:n]
    # Удалим средние — стандартная подготовка
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()

    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-12)
    s_target = alpha * reference
    e_dist = estimate - s_target
    return float(10.0 * np.log10(
        (np.sum(s_target ** 2) + 1e-12) / (np.sum(e_dist ** 2) + 1e-12)
    ))
