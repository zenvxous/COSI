import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, firwin, butter, filtfilt
import matplotlib.pyplot as plt

N = int(input("Введите длину сигнала N (степень двойки): "))
if (N & (N - 1)) != 0:
    raise ValueError("Ошибка: N не является степенью двойки!")

fs = 8000
t = np.arange(N) / fs

f0 = 220
signal = np.zeros(N)
amplitudes = [1.0, 0.5, 0.3, 0.2, 0.1]
for k, amp in enumerate(amplitudes, start=1):
    signal += amp * np.sin(2 * np.pi * f0 * k * t)
signal += 0.05 * np.random.randn(N)

M = 16
b_mavg = np.ones(M) / M
a_mavg = 1
signal_mavg = lfilter(b_mavg, a_mavg, signal)

fc = 300
M_fir = 101
b_fir = firwin(M_fir, cutoff=fc, window='hamming', fs=fs)
a_fir = 1
signal_lowpass = lfilter(b_fir, a_fir, signal)

f_center = 275
BW = 250
low_freq = f_center - BW/2
high_freq = f_center + BW/2
order = 4
b_iir, a_iir = butter(order, [low_freq, high_freq], btype='bandpass', fs=fs)
signal_bandpass = filtfilt(b_iir, a_iir, signal)

wavfile.write("output_moving_average.wav", fs, 
              np.int16(signal_mavg/np.max(np.abs(signal_mavg)) * 32767))
wavfile.write("output_lowpass_fir.wav", fs, 
              np.int16(signal_lowpass/np.max(np.abs(signal_lowpass)) * 32767))
wavfile.write("output_bandpass_iir.wav", fs, 
              np.int16(signal_bandpass/np.max(np.abs(signal_bandpass)) * 32767))

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