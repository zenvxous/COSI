import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from scipy.io import wavfile  

def generate_signals(N):
    fs = 4000
    t = np.arange(N) / fs
    
    f0_x = 220
    x = (1.0 * np.sin(2 * np.pi * 1 * f0_x * t) + 
         0.3 * np.sin(2 * np.pi * 4 * f0_x * t) + 
         0.1 * np.sin(2 * np.pi * 6 * f0_x * t))

    f0_y = 220
    y = (1.0 * np.sin(2 * np.pi * 1 * f0_y * t) + 
         0.4 * np.sin(2 * np.pi * 3 * f0_y * t) + 
         0.2 * np.sin(2 * np.pi * 5 * f0_y * t))
    
    return x, y, t, fs

def next_power_of_2(n):
    return 1 if n == 0 else 2**(n - 1).bit_length()

def clean_phase(spectrum, threshold=1e-10):
    phase = np.angle(spectrum)
    magnitude = np.abs(spectrum)
    phase[magnitude < threshold] = 0
    return phase

def manual_dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def manual_idft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    return x / N

def manual_fft_recursive(x):
    N = len(x)
    if N <= 1: return x
    even = manual_fft_recursive(x[0::2])
    odd = manual_fft_recursive(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return np.array([even[k] + T[k] for k in range(N // 2)] + \
                    [even[k] - T[k] for k in range(N // 2)])

def manual_ifft(X):
    N = len(X)
    X_conj = np.conjugate(X)
    res = manual_fft_recursive(X_conj)
    return np.conjugate(res) / N

def manual_convolution(x, y):
    N, M = len(x), len(y)
    total_len = N + M - 1
    result = np.zeros(total_len)
    for n in range(total_len):
        val = 0
        for k in range(N):
            if 0 <= n - k < M:
                val += x[k] * y[n - k]
        result[n] = val
    return result

def manual_correlation(x, y):
    return manual_convolution(x, y[::-1])

def run_lab():
    
    try:
        raw_input = input("Введите количество отсчетов N (обязательно степень 2, например 64, 128, 256, 8192): ")
        N = int(raw_input)
        if N <= 0 or (N & (N - 1) != 0):
            print("Внимание! Ваш ручной алгоритм БПФ требует N равное степени двойки (2, 4, 8, ...).")
            choice = input("Продолжить? (y/n): ")
            if choice.lower() != 'y':
                sys.exit()
    except ValueError:
        print("Ошибка: введите целое число.")
        sys.exit()

    x, y, t, fs = generate_signals(N)
    print(f"\nСгенерированы сигналы длиной N={N}, fs={fs} Гц")

    print("Сохранение сигналов в WAV файлы...", end=" ", flush=True)
    x_wav = np.int16((x / np.max(np.abs(x))) * 32767)
    y_wav = np.int16((y / np.max(np.abs(y))) * 32767)
    
    wavfile.write("signal_x_strings.wav", fs, x_wav)
    wavfile.write("signal_y_winds.wav", fs, y_wav)
    print("Готово. (Созданы 'signal_x_strings.wav' и 'signal_y_winds.wav')")

    print("1. Расчет ДПФ", end=" ", flush=True)
    t0 = time.perf_counter()
    dft_x = manual_dft(x);      idft_x = np.real(manual_idft(dft_x))
    dft_y = manual_dft(y);      idft_y = np.real(manual_idft(dft_y))
    t_dft = time.perf_counter() - t0
    print(f"Готово. Время: {t_dft:.5f} сек")

    print("2. Расчет БПФ", end=" ", flush=True)
    t0 = time.perf_counter()
    fft_x = manual_fft_recursive(x); ifft_x = np.real(manual_ifft(fft_x))
    fft_y = manual_fft_recursive(y); ifft_y = np.real(manual_ifft(fft_y))
    t_fft = time.perf_counter() - t0
    print(f"Готово. Время: {t_fft:.5f} сек")

    print("3. Расчет свертки", end=" ", flush=True)
    t0 = time.perf_counter()
    conv_man = manual_convolution(x, y)
    t_conv = time.perf_counter() - t0
    print(f"   Готово. Время: {t_conv:.5f} сек")

    print("4. Расчет корреляции", end=" ", flush=True)
    t0 = time.perf_counter()
    corr_man = manual_correlation(x, y)
    t_corr = time.perf_counter() - t0
    print(f"Готово. Время: {t_corr:.5f} сек")

    print("5. Расчет теорем", end=" ", flush=True)
    t0 = time.perf_counter()
    L = len(x) + len(y) - 1
    N_fft = next_power_of_2(L)
    x_pad = np.pad(x, (0, N_fft - len(x)))
    y_pad = np.pad(y, (0, N_fft - len(y)))
    y_rev_pad = np.pad(y[::-1], (0, N_fft - len(y)))

    fft_x_pad = manual_fft_recursive(x_pad)
    fft_y_pad = manual_fft_recursive(y_pad)
    fft_y_rev_pad = manual_fft_recursive(y_rev_pad)

    spec_conv = fft_x_pad * fft_y_pad
    conv_fft_man = np.real(manual_ifft(spec_conv)[:L])

    spec_corr = fft_x_pad * fft_y_rev_pad
    corr_fft_man = np.real(manual_ifft(spec_corr)[:L])
    t_theo = time.perf_counter() - t0
    print(f"Готово. Время: {t_theo:.5f} сек")

    print("6. Расчет библиотечных функций", end=" ", flush=True)
    t0 = time.perf_counter()
    lib_fft_x = np.fft.fft(x)
    lib_fft_y = np.fft.fft(y)
    lib_conv = np.convolve(x, y, mode='full')
    lib_corr = np.correlate(x, y, mode='full')
    t_lib = time.perf_counter() - t0
    print(f"   Готово. Время: {t_lib:.5f} сек")

    print(f"ДПФ (O(N^2)) заняло: {t_dft:.5f} сек")
    print(f"БПФ (O(N log2 N)) заняло: {t_fft:.5f} сек")
    print(f"БПФ быстрее ДПФ в {t_dft/t_fft:.1f} раз")

    print("\nГенерация графиков...")
    freqs = np.fft.fftfreq(N, 1/fs)
    mask = freqs >= 0
    f_pos = freqs[mask]
    limit_f = 1500 

    plt.figure(figsize=(12, 16))
    plt.suptitle(f"Стр. 1: Анализ x(t) (N={N})", fontsize=16)
    
    ax1 = plt.subplot(4, 2, 1); ax1.plot(t, x, 'b'); ax1.set_title("1. x(t) (Струнные)")
    ax1.grid()
    
    ax3 = plt.subplot(4, 2, 3); ax3.stem(f_pos, np.abs(dft_x)[mask])
    ax3.set_title("3. ДПФ: амплитудный спектр"); ax3.set_xlim(0, limit_f); ax3.grid()
    ax4 = plt.subplot(4, 2, 4); ax4.stem(f_pos, clean_phase(dft_x)[mask])
    ax4.set_title("4. ДПФ: фазовый спектр"); ax4.set_xlim(0, limit_f); ax4.grid()
    ax5 = plt.subplot(4, 2, 5); ax5.plot(t, idft_x, 'g--'); ax5.plot(t, x, 'b:', alpha=0.5)
    ax5.set_title("5. ОДПФ (Восстановление)"); ax5.grid()

    ax6 = plt.subplot(4, 2, 6); ax6.stem(f_pos, np.abs(fft_x)[mask])
    ax6.set_title("6. БПФ: амплитудный спектр"); ax6.set_xlim(0, limit_f); ax6.grid()
    ax7 = plt.subplot(4, 2, 7); ax7.stem(f_pos, clean_phase(fft_x)[mask])
    ax7.set_title("7. БПФ: фазовый спектр"); ax7.set_xlim(0, limit_f); ax7.grid()
    ax8 = plt.subplot(4, 2, 8); ax8.plot(t, ifft_x, 'r--'); ax8.plot(t, x, 'b:', alpha=0.5)
    ax8.set_title("8. ОБПФ (Восстановление)"); ax8.grid()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Page1_X.png")

    plt.figure(figsize=(12, 16))
    plt.suptitle(f"Стр. 2: Анализ y(t) (N={N})", fontsize=16)
    
    ax2 = plt.subplot(4, 2, 1); ax2.plot(t, y, 'orange'); ax2.set_title("2. y(t) (Духовые)")
    ax2.grid()

    ax9 = plt.subplot(4, 2, 3); ax9.stem(f_pos, np.abs(dft_y)[mask], linefmt='C1-')
    ax9.set_title("9. ДПФ: амплитудный спектр"); ax9.set_xlim(0, limit_f); ax9.grid()
    ax10 = plt.subplot(4, 2, 4); ax10.stem(f_pos, clean_phase(dft_y)[mask], linefmt='C1-')
    ax10.set_title("10. ДПФ: фазовый спектр"); ax10.set_xlim(0, limit_f); ax10.grid()
    ax11 = plt.subplot(4, 2, 5); ax11.plot(t, idft_y, 'g--'); ax11.plot(t, y, 'orange', alpha=0.3)
    ax11.set_title("11. ОДПФ (Восстановление)"); ax11.grid()

    ax12 = plt.subplot(4, 2, 6); ax12.stem(f_pos, np.abs(fft_y)[mask], linefmt='C1-')
    ax12.set_title("12. БПФ: амплитудный спектр"); ax12.set_xlim(0, limit_f); ax12.grid()
    ax13 = plt.subplot(4, 2, 7); ax13.stem(f_pos, clean_phase(fft_y)[mask], linefmt='C1-')
    ax13.set_title("13. БПФ: фазовый спектр"); ax13.set_xlim(0, limit_f); ax13.grid()
    ax14 = plt.subplot(4, 2, 8); ax14.plot(t, ifft_y, 'r--'); ax14.plot(t, y, 'orange', alpha=0.3)
    ax14.set_title("14. ОБПФ (Восстановление)"); ax14.grid()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Page2_Y.png")

    plt.figure(figsize=(12, 10))
    plt.suptitle("Стр. 3: Ручная реализация операций", fontsize=16)

    ax15 = plt.subplot(2, 2, 1); ax15.plot(conv_man, 'k')
    ax15.set_title("15. Свертка (Формула)"); ax15.grid()

    ax16 = plt.subplot(2, 2, 2); ax16.plot(conv_fft_man, 'r--')
    ax16.set_title("16. Свертка через БПФ"); ax16.grid()

    ax17 = plt.subplot(2, 2, 3); ax17.plot(corr_man, 'k')
    ax17.set_title("17. Корреляция (Формула)"); ax17.grid()

    ax18 = plt.subplot(2, 2, 4); ax18.plot(corr_fft_man, 'g--')
    ax18.set_title("18. Корреляция через БПФ"); ax18.grid()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Page3_Ops_Manual.png")

    plt.figure(figsize=(12, 12))
    plt.suptitle("Стр. 4: Open Source библиотеки", fontsize=16)

    ax19 = plt.subplot(3, 2, 1); ax19.stem(f_pos, np.abs(lib_fft_x)[mask])
    ax19.set_title("19. x(t) Lib БПФ Амплитуда"); ax19.set_xlim(0, limit_f); ax19.grid()

    ax20 = plt.subplot(3, 2, 2); ax20.stem(f_pos, clean_phase(lib_fft_x)[mask])
    ax20.set_title("20. x(t) Lib БПФ Фаза"); ax20.set_xlim(0, limit_f); ax20.grid()

    ax21 = plt.subplot(3, 2, 3); ax21.stem(f_pos, np.abs(lib_fft_y)[mask], linefmt='C1-')
    ax21.set_title("21. y(t) Lib БПФ Амплитуда"); ax21.set_xlim(0, limit_f); ax21.grid()

    ax22 = plt.subplot(3, 2, 4); ax22.stem(f_pos, clean_phase(lib_fft_y)[mask], linefmt='C1-')
    ax22.set_title("22. y(t) Lib БПФ Фаза"); ax22.set_xlim(0, limit_f); ax22.grid()

    ax23 = plt.subplot(3, 2, 5); ax23.plot(lib_conv, 'm')
    ax23.set_title("23. Свертка (Lib)"); ax23.grid()

    ax24 = plt.subplot(3, 2, 6); ax24.plot(lib_corr, 'c')
    ax24.set_title("24. Корреляция (Lib)"); ax24.grid()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Page4_Lib.png")
    
    print("Готово. Создано 4 файла графиков.")

if __name__ == "__main__":
    run_lab()