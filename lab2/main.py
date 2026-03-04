import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, firwin, butter, filtfilt
import matplotlib.pyplot as plt

# Ввод длины сигнала N (степень двойки)
N = int(input("Введите длину сигнала N (степень двойки): "))
if (N & (N - 1)) != 0:
    raise ValueError("Ошибка: N не является степенью двойки!")

fs = 8000  # Частота дискретизации (Гц)
t = np.arange(N) / fs  # Временная ось

# 1. Генерация синтетического сигнала (струнного типа)
f0 = 220  # Основная частота 220 Гц
signal = np.zeros(N)
# Суммируем первые 5 гармоник с убывающими амплитудами
amplitudes = [1.0, 0.5, 0.3, 0.2, 0.1]
for k, amp in enumerate(amplitudes, start=1):
    signal += amp * np.sin(2 * np.pi * f0 * k * t)
# Добавляем белый шум небольшого уровня
signal += 0.05 * np.random.randn(N)

# 2. Применение фильтров

# 2.1. Однородный скользящий средний фильтр (M = 16)
M = 16
b_mavg = np.ones(M) / M
a_mavg = 1
signal_mavg = lfilter(b_mavg, a_mavg, signal)

# 2.2. НЧ FIR-фильтр (Hamming, M = 101, fc = 300 Гц)
fc = 300  # частота среза 300 Гц
M_fir = 101
b_fir = firwin(M_fir, cutoff=fc, window='hamming', fs=fs)
a_fir = 1
signal_lowpass = lfilter(b_fir, a_fir, signal)

# 2.3. Полосовой IIR-фильтр (Butterworth, f0 = 275 Гц, полоса = 250 Гц)
f_center = 275  # центральная частота
BW = 250  # ширина полосы
low_freq = f_center - BW/2  # нижняя граничная частота
high_freq = f_center + BW/2 # верхняя граничная частота
order = 4  # порядок фильтра
b_iir, a_iir = butter(order, [low_freq, high_freq], btype='bandpass', fs=fs)
signal_bandpass = filtfilt(b_iir, a_iir, signal)  # zero-phase фильтрация

# 3. Сохранение отфильтрованных сигналов в WAV-файлы (номиналируем к int16)
wavfile.write("output_moving_average.wav", fs, 
              np.int16(signal_mavg/np.max(np.abs(signal_mavg)) * 32767))
wavfile.write("output_lowpass_fir.wav", fs, 
              np.int16(signal_lowpass/np.max(np.abs(signal_lowpass)) * 32767))
wavfile.write("output_bandpass_iir.wav", fs, 
              np.int16(signal_bandpass/np.max(np.abs(signal_bandpass)) * 32767))

# 4. Построение графиков

# 4.1. Временные области сигналов
plt.figure(figsize=(10, 8))
plt.subplot(4, 1, 1)
plt.plot(t, signal)
plt.title("Исходный сигнал (временная область)")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, signal_mavg, color='orange')
plt.title("Скользящее среднее (M=16)")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, signal_lowpass, color='green')
plt.title("НЧ FIR-фильтр (fc = 300 Гц)")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, signal_bandpass, color='red')
plt.title("Полосовой IIR-фильтр (f0 = 275 Гц, BW = 250 Гц)")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда")
plt.grid(True)

plt.tight_layout()
plt.savefig("waveform_plot.png")
plt.close()

# 4.2. Амплитудные спектры сигналов
def plot_spectrum(x, fs, title, ax):
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    ax.plot(freqs, np.abs(X))
    ax.set_title(title)
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("Амплитуда")
    ax.grid(True)

plt.figure(figsize=(10, 8))
ax1 = plt.subplot(4, 1, 1)
plot_spectrum(signal, fs, "Спектр исходного сигнала", ax1)
ax2 = plt.subplot(4, 1, 2)
plot_spectrum(signal_mavg, fs, "Спектр (скользящее среднее)", ax2)
ax3 = plt.subplot(4, 1, 3)
plot_spectrum(signal_lowpass, fs, "Спектр (НЧ FIR)", ax3)
ax4 = plt.subplot(4, 1, 4)
plot_spectrum(signal_bandpass, fs, "Спектр (Полосовой IIR)", ax4)

plt.tight_layout()
plt.savefig("spectrum_plot.png")
plt.close()

print("Готово. Сигналы сохранены и графики построены.")