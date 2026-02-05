import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# =========================================================
# 1. ГЕНЕРАЦИЯ СИГНАЛОВ (ВАРИАНТ 4)
# =========================================================
def generate_signals(N):
    # fs=4000 Гц, чтобы покрыть гармоники до 1320 Гц
    fs = 4000
    t = np.arange(N) / fs
    
    # x(t): Струнные (f0=220, гармоники 1, 4, 6)
    f0_x = 220
    x = (1.0 * np.sin(2 * np.pi * 1 * f0_x * t) + 
         0.3 * np.sin(2 * np.pi * 4 * f0_x * t) + 
         0.1 * np.sin(2 * np.pi * 6 * f0_x * t))

    # y(t): Духовые (f0=220, гармоники 1, 3, 5)
    f0_y = 220
    y = (1.0 * np.sin(2 * np.pi * 1 * f0_y * t) + 
         0.4 * np.sin(2 * np.pi * 3 * f0_y * t) + 
         0.2 * np.sin(2 * np.pi * 5 * f0_y * t))
    
    return x, y, t, fs

def next_power_of_2(n):
    return 1 if n == 0 else 2**(n - 1).bit_length()

def clean_phase(spectrum, threshold=1e-10):
    """Убирает шум фазы там, где амплитуда почти 0"""
    phase = np.angle(spectrum)
    magnitude = np.abs(spectrum)
    phase[magnitude < threshold] = 0
    return phase

# =========================================================
# 2. РУЧНЫЕ РЕАЛИЗАЦИИ (MANUAL)
# =========================================================

# --- ДПФ (DFT) O(N^2) ---
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

# --- БПФ (FFT) O(N log N) ---
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

# --- Свертка и Корреляция ---
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

# =========================================================
# 3. ОСНОВНОЙ КОД
# =========================================================
def run_lab():
    print("=== ЛАБОРАТОРНАЯ РАБОТА №1 (ВАРИАНТ 4) ===")
    
    # 1. Ввод N
    try:
        raw_input = input("Введите количество отсчетов N (обязательно степень 2, например 64, 128, 256): ")
        N = int(raw_input)
        # Проверка на степень двойки (битовая магия: если N & (N-1) == 0, то это степень 2)
        if N <= 0 or (N & (N - 1) != 0):
            print("Внимание! Ваш ручной алгоритм БПФ требует N равное степени двойки (2, 4, 8, ...).")
            print("Программа может упасть или работать некорректно.")
            choice = input("Продолжить? (y/n): ")
            if choice.lower() != 'y':
                sys.exit()
    except ValueError:
        print("Ошибка: введите целое число.")
        sys.exit()

    # Генерация
    x, y, t, fs = generate_signals(N)
    print(f"\nСгенерированы сигналы длиной N={N}, fs={fs} Гц")

    print("-" * 40)
    print("НАЧАЛО ВЫЧИСЛЕНИЙ (ОЦЕНКА ЭФФЕКТИВНОСТИ)")
    print("-" * 40)

    # --- РУЧНЫЕ РАСЧЕТЫ ---
    print("1. Считаем ДПФ (DFT) Manual...", end=" ", flush=True)
    t0 = time.perf_counter()
    dft_x = manual_dft(x);      idft_x = np.real(manual_idft(dft_x))
    dft_y = manual_dft(y);      idft_y = np.real(manual_idft(dft_y))
    t_dft = time.perf_counter() - t0
    print(f"Готово. Время: {t_dft:.5f} сек")

    print("2. Считаем БПФ (FFT) Manual...", end=" ", flush=True)
    t0 = time.perf_counter()
    fft_x = manual_fft_recursive(x); ifft_x = np.real(manual_ifft(fft_x))
    fft_y = manual_fft_recursive(y); ifft_y = np.real(manual_ifft(fft_y))
    t_fft = time.perf_counter() - t0
    print(f"Готово. Время: {t_fft:.5f} сек")

    print("3. Считаем Свертку Manual...", end=" ", flush=True)
    t0 = time.perf_counter()
    conv_man = manual_convolution(x, y)
    t_conv = time.perf_counter() - t0
    print(f"   Готово. Время: {t_conv:.5f} сек")

    print("4. Считаем Корреляцию Manual...", end=" ", flush=True)
    t0 = time.perf_counter()
    corr_man = manual_correlation(x, y)
    t_corr = time.perf_counter() - t0
    print(f"Готово. Время: {t_corr:.5f} сек")

    # --- ТЕОРЕМЫ (ЧЕРЕЗ БПФ) ---
    print("5. Считаем Теоремы (через FFT Manual)...", end=" ", flush=True)
    t0 = time.perf_counter()
    L = len(x) + len(y) - 1
    N_fft = next_power_of_2(L)
    x_pad = np.pad(x, (0, N_fft - len(x)))
    y_pad = np.pad(y, (0, N_fft - len(y)))
    y_rev_pad = np.pad(y[::-1], (0, N_fft - len(y)))

    # Используем ручной FFT для "честности"
    fft_x_pad = manual_fft_recursive(x_pad)
    fft_y_pad = manual_fft_recursive(y_pad)
    fft_y_rev_pad = manual_fft_recursive(y_rev_pad)

    # Свертка через БПФ
    spec_conv = fft_x_pad * fft_y_pad
    conv_fft_man = np.real(manual_ifft(spec_conv)[:L])

    # Корреляция через БПФ
    spec_corr = fft_x_pad * fft_y_rev_pad
    corr_fft_man = np.real(manual_ifft(spec_corr)[:L])
    t_theo = time.perf_counter() - t0
    print(f"Готово. Время: {t_theo:.5f} сек")

    # --- БИБЛИОТЕКИ ---
    print("6. Считаем Библиотечные функции...", end=" ", flush=True)
    t0 = time.perf_counter()
    lib_fft_x = np.fft.fft(x)
    lib_fft_y = np.fft.fft(y)
    lib_conv = np.convolve(x, y, mode='full')
    lib_corr = np.correlate(x, y, mode='full')
    t_lib = time.perf_counter() - t0
    print(f"   Готово. Время: {t_lib:.5f} сек")

    print("-" * 40)
    print("ВЫВОД ЭФФЕКТИВНОСТИ:")
    print(f"ДПФ (O(N^2)) заняло: {t_dft:.5f} сек")
    print(f"БПФ (O(N log N)) заняло: {t_fft:.5f} сек")
    print(f"БПФ быстрее ДПФ в {t_dft/t_fft:.1f} раз")
    print("-" * 40)

    # =========================================================
    # 4. ОТРИСОВКА (4 СТРАНИЦЫ, 24 ГРАФИКА)
    # =========================================================
    print("Генерация графиков (4 страницы)...")
    freqs = np.fft.fftfreq(N, 1/fs)
    mask = freqs >= 0
    f_pos = freqs[mask]
    limit_f = 1500 

    # --- СТРАНИЦА 1: X(t) Analysis (8 графиков) ---
    plt.figure(figsize=(12, 16))
    plt.suptitle(f"Стр. 1: Анализ x(t) (N={N})", fontsize=16)
    
    # 1. x(t)
    ax1 = plt.subplot(4, 2, 1); ax1.plot(t, x, 'b'); ax1.set_title("1. x(t) (Струнные)")
    ax1.grid()
    
    # 3-5. ДПФ
    ax3 = plt.subplot(4, 2, 3); ax3.stem(f_pos, np.abs(dft_x)[mask])
    ax3.set_title("3. ДПФ: амплитудный спектр"); ax3.set_xlim(0, limit_f); ax3.grid()
    ax4 = plt.subplot(4, 2, 4); ax4.stem(f_pos, clean_phase(dft_x)[mask])
    ax4.set_title("4. ДПФ: фазовый спектр"); ax4.set_xlim(0, limit_f); ax4.grid()
    ax5 = plt.subplot(4, 2, 5); ax5.plot(t, idft_x, 'g--'); ax5.plot(t, x, 'b:', alpha=0.5)
    ax5.set_title("5. ОДПФ (Восстановление)"); ax5.grid()

    # 6-8. БПФ
    ax6 = plt.subplot(4, 2, 6); ax6.stem(f_pos, np.abs(fft_x)[mask])
    ax6.set_title("6. БПФ: амплитудный спектр"); ax6.set_xlim(0, limit_f); ax6.grid()
    ax7 = plt.subplot(4, 2, 7); ax7.stem(f_pos, clean_phase(fft_x)[mask])
    ax7.set_title("7. БПФ: фазовый спектр"); ax7.set_xlim(0, limit_f); ax7.grid()
    ax8 = plt.subplot(4, 2, 8); ax8.plot(t, ifft_x, 'r--'); ax8.plot(t, x, 'b:', alpha=0.5)
    ax8.set_title("8. ОБПФ (Восстановление)"); ax8.grid()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Page1_X.png")

    # --- СТРАНИЦА 2: Y(t) Analysis (8 графиков) ---
    plt.figure(figsize=(12, 16))
    plt.suptitle(f"Стр. 2: Анализ y(t) (N={N})", fontsize=16)
    
    # 2. y(t)
    ax2 = plt.subplot(4, 2, 1); ax2.plot(t, y, 'orange'); ax2.set_title("2. y(t) (Духовые)")
    ax2.grid()

    # 9-11. ДПФ
    ax9 = plt.subplot(4, 2, 3); ax9.stem(f_pos, np.abs(dft_y)[mask], linefmt='C1-')
    ax9.set_title("9. ДПФ: амплитудный спектр"); ax9.set_xlim(0, limit_f); ax9.grid()
    ax10 = plt.subplot(4, 2, 4); ax10.stem(f_pos, clean_phase(dft_y)[mask], linefmt='C1-')
    ax10.set_title("10. ДПФ: фазовый спектр"); ax10.set_xlim(0, limit_f); ax10.grid()
    ax11 = plt.subplot(4, 2, 5); ax11.plot(t, idft_y, 'g--'); ax11.plot(t, y, 'orange', alpha=0.3)
    ax11.set_title("11. ОДПФ (Восстановление)"); ax11.grid()

    # 12-14. БПФ
    ax12 = plt.subplot(4, 2, 6); ax12.stem(f_pos, np.abs(fft_y)[mask], linefmt='C1-')
    ax12.set_title("12. БПФ: амплитудный спектр"); ax12.set_xlim(0, limit_f); ax12.grid()
    ax13 = plt.subplot(4, 2, 7); ax13.stem(f_pos, clean_phase(fft_y)[mask], linefmt='C1-')
    ax13.set_title("13. БПФ: фазовый спектр"); ax13.set_xlim(0, limit_f); ax13.grid()
    ax14 = plt.subplot(4, 2, 8); ax14.plot(t, ifft_y, 'r--'); ax14.plot(t, y, 'orange', alpha=0.3)
    ax14.set_title("14. ОБПФ (Восстановление)"); ax14.grid()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Page2_Y.png")

    # --- СТРАНИЦА 3: Операции Manual (4 графика) ---
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

    # --- СТРАНИЦА 4: Библиотеки (6 графиков) ---
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