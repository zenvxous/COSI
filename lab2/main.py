import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter, firwin, butter
import sys

def generate_signal(N):
    fs = 4000
    t = np.arange(N) / fs
    f0 = 220
    x = (1.0 * np.sin(2*np.pi*1*f0*t) +
         0.3 * np.sin(2*np.pi*4*f0*t) +
         0.1 * np.sin(2*np.pi*6*f0*t))
    noise = 0.05 * np.random.randn(N)
    x = x + noise
    return x, t, fs

def save_wav(name, signal, fs):
    signal_norm = signal / np.max(np.abs(signal))
    signal_int16 = np.int16(signal_norm * 32767)
    wavfile.write(name, fs, signal_int16)

def get_spectrum(signal, fs):
    N = len(signal)
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    mask = freqs >= 0
    return freqs[mask], np.abs(spectrum)[mask]

def run_lab():
    try:
        raw = input("Введите длину сигнала N (степень 2): ")
        N = int(raw)
        if N <= 0 or (N & (N-1)) != 0:
            print("N должен быть степенью 2")
            sys.exit()
    except:
        print("Ошибка ввода")
        sys.exit()

    x, t, fs = generate_signal(N)

    save_wav("input_signal.wav", x, fs)

    M = 16
    b_ma = np.ones(M) / M
    a_ma = [1]
    y_ma = lfilter(b_ma, a_ma, x)
    save_wav("output_moving_average.wav", y_ma, fs)

    fc = 300
    fir_coeff = firwin(
        numtaps=101,
        cutoff=fc,
        window='hamming',
        fs=fs
    )
    y_fir = lfilter(fir_coeff, 1.0, x)
    save_wav("output_lowpass_fir.wav", y_fir, fs)

    f0 = 275
    BW = 250
    low = (f0 - BW/2) / (fs/2)
    high = (f0 + BW/2) / (fs/2)
    b_iir, a_iir = butter(4, [low, high], btype='band')
    y_iir = lfilter(b_iir, a_iir, x)
    save_wav("output_bandpass_iir.wav", y_iir, fs)

    f_x, sp_x = get_spectrum(x, fs)
    f_ma, sp_ma = get_spectrum(y_ma, fs)
    f_fir, sp_fir = get_spectrum(y_fir, fs)
    f_iir, sp_iir = get_spectrum(y_iir, fs)

    plt.figure(figsize=(12,10))

    plt.subplot(4,1,1)
    plt.plot(t, x)
    plt.title("Исходный сигнал")
    plt.grid()

    plt.subplot(4,1,2)
    plt.plot(t, y_ma)
    plt.title("После рекурсивного фильтра (M=16)")
    plt.grid()

    plt.subplot(4,1,3)
    plt.plot(t, y_fir)
    plt.title("После FIR НЧ фильтра")
    plt.grid()

    plt.subplot(4,1,4)
    plt.plot(t, y_iir)
    plt.title("После IIR полосового фильтра")
    plt.grid()

    plt.tight_layout()
    plt.savefig("waveform_plot.png")

    plt.figure(figsize=(12,10))

    plt.subplot(4,1,1)
    plt.stem(f_x, sp_x)
    plt.title("Спектр исходного сигнала")
    plt.xlim(0,1500)
    plt.grid()

    plt.subplot(4,1,2)
    plt.stem(f_ma, sp_ma)
    plt.title("Спектр после рекурсивного фильтра")
    plt.xlim(0,1500)
    plt.grid()

    plt.subplot(4,1,3)
    plt.stem(f_fir, sp_fir)
    plt.title("Спектр после FIR фильтра")
    plt.xlim(0,1500)
    plt.grid()

    plt.subplot(4,1,4)
    plt.stem(f_iir, sp_iir)
    plt.title("Спектр после IIR фильтра")
    plt.xlim(0,1500)
    plt.grid()

    plt.tight_layout()
    plt.savefig("spectrum_plot.png")

    print("\nФайлы созданы:")
    print("input_signal.wav")
    print("output_moving_average.wav")
    print("output_lowpass_fir.wav")
    print("output_bandpass_iir.wav")
    print("waveform_plot.png")
    print("spectrum_plot.png")

if __name__ == "__main__":
    run_lab()