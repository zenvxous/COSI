import numpy as np
import matplotlib.pyplot as plt
import time

# =========================================================
# 1. ГЕНЕРАЦИЯ (Вариант 4)
# =========================================================
def generate_signals(N=128):
    fs = 4000  # Частота дискретизации
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

# =========================================================
# 2. РУЧНЫЕ РЕАЛИЗАЦИИ
# =========================================================
def manual_fft_recursive(x):
    N = len(x)
    if N <= 1: return x
    even = manual_fft_recursive(x[0::2])
    odd = manual_fft_recursive(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return np.array([even[k] + T[k] for k in range(N // 2)] + \
                    [even[k] - T[k] for k in range(N // 2)])

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
    # Корреляция = свертка x с перевернутым y
    return manual_convolution(x, y[::-1])

# Вспомогательная функция для очистки фазы (убирает шум)
def clean_phase(spectrum, threshold=1e-10):
    # Если амплитуда слишком мала, фаза не имеет смысла (ставим 0)
    phase = np.angle(spectrum)
    magnitude = np.abs(spectrum)
    phase[magnitude < threshold] = 0
    return phase

# =========================================================
# 3. ОСНОВНОЙ СКРИПТ
# =========================================================
def run_lab():
    x, y, t, fs = generate_signals(N=128)
    
    print(f"=== ЛАБОРАТОРНАЯ РАБОТА №1 (Вариант 4) ===")
    print(f"fs = {fs} Гц, N = {len(x)}")

    # --- 1. FFT и Спектры X и Y ---
    print("\n[1] Расчет спектров сигналов")
    t0 = time.perf_counter()
    spec_x = manual_fft_recursive(x) # Ручной FFT
    spec_y = np.fft.fft(y)           # Библиотечный для Y
    print(f"  FFT calc time: {time.perf_counter()-t0:.6f} s")

    # --- 2. Свертка ---
    print("\n[2] Свертка")
    # Ручная
    t0 = time.perf_counter()
    conv_man = manual_convolution(x, y)
    print(f"  Manual time: {time.perf_counter()-t0:.6f} s")
    
    # Теорема (FFT)
    L = len(x) + len(y) - 1
    N_fft = next_power_of_2(L)
    x_pad = np.pad(x, (0, N_fft - len(x)))
    y_pad = np.pad(y, (0, N_fft - len(y)))
    
    t0 = time.perf_counter()
    spec_conv_complex = np.fft.fft(x_pad) * np.fft.fft(y_pad) # Спектр свертки (комплексный)
    conv_theo = np.real(np.fft.ifft(spec_conv_complex)[:L])
    print(f"  FFT Theorem time: {time.perf_counter()-t0:.6f} s")
    
    # --- 3. Спектр Результата Свертки ---
    # По заданию нужно построить спектр результата свертки
    # Мы можем взять FFT от результата свертки (conv_theo)
    # Или использовать уже посчитанный spec_conv_complex (но он для дополненного нулями)
    # Лучше взять честный FFT от полученного временного сигнала, чтобы показать "Спектр результата"
    spec_res_conv = np.fft.fft(conv_theo)

    # --- 4. Корреляция ---
    print("\n[3] Корреляция")
    t0 = time.perf_counter()
    corr_man = manual_correlation(x, y)
    print(f"  Manual time: {time.perf_counter()-t0:.6f} s")
    
    y_rev_pad = np.pad(y[::-1], (0, N_fft - len(y)))
    corr_theo = np.real(np.fft.ifft(np.fft.fft(x_pad) * np.fft.fft(y_rev_pad))[:L])

    # --- 5. Визуализация ---
    print("\n[INFO] Генерация графиков...")
    plot_signals_and_spectra(t, x, y, spec_x, spec_y, fs)
    plot_operations(conv_man, conv_theo, spec_res_conv, corr_man, corr_theo, fs)
    print("[DONE] Графики сохранены: 'page1_signals.png' и 'page2_ops.png'")

def plot_signals_and_spectra(t, x, y, X, Y, fs):
    """Страница 1: Исходные сигналы и их спектры"""
    plt.figure(figsize=(14, 12))
    plt.suptitle("Часть 1: Анализ исходных сигналов", fontsize=16)
    
    # Временная область
    plt.subplot(3, 2, 1)
    plt.plot(t, x, 'b')
    plt.title("Сигнал x(t): Струнные")
    plt.grid(True)
    
    plt.subplot(3, 2, 2)
    plt.plot(t, y, 'orange')
    plt.title("Сигнал y(t): Духовые")
    plt.grid(True)

    # Частотная ось
    freqs_x = np.fft.fftfreq(len(x), 1/fs)
    mask_x = freqs_x >= 0 # Только положительные частоты
    
    # Амплитудные спектры
    plt.subplot(3, 2, 3)
    plt.stem(freqs_x[mask_x], np.abs(X)[mask_x])
    plt.title("Амплитудный спектр |X(f)|")
    plt.xlim(0, 1500); plt.grid(True)
    
    plt.subplot(3, 2, 4)
    plt.stem(freqs_x[mask_x], np.abs(Y)[mask_x], linefmt='C1-', markerfmt='C1o')
    plt.title("Амплитудный спектр |Y(f)|")
    plt.xlim(0, 1500); plt.grid(True)
    
    # Фазовые спектры
    plt.subplot(3, 2, 5)
    plt.stem(freqs_x[mask_x], clean_phase(X)[mask_x])
    plt.title("Фазовый спектр arg(X(f))")
    plt.xlim(0, 1500); plt.grid(True)
    
    plt.subplot(3, 2, 6)
    plt.stem(freqs_x[mask_x], clean_phase(Y)[mask_x], linefmt='C1-', markerfmt='C1o')
    plt.title("Фазовый спектр arg(Y(f))")
    plt.xlim(0, 1500); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('page1_signals.png')

def plot_operations(conv_ref, conv_fft, spec_conv, corr_ref, corr_fft, fs):
    """Страница 2: Результаты операций и спектр свертки"""
    plt.figure(figsize=(14, 12))
    plt.suptitle("Часть 2: Свертка, Корреляция и их Спектральный анализ", fontsize=16)
    
    # 1. Временная область СВЕРТКА
    plt.subplot(3, 2, 1)
    plt.plot(conv_ref, 'k', lw=3, alpha=0.3, label='Ручная')
    plt.plot(conv_fft, 'r--', label='FFT Теорема')
    plt.title("Результат Свертки (Временная область)")
    plt.legend(); plt.grid(True)
    
    # 2. Временная область КОРРЕЛЯЦИЯ
    plt.subplot(3, 2, 2)
    plt.plot(corr_ref, 'k', lw=3, alpha=0.3, label='Ручная')
    plt.plot(corr_fft, 'g--', label='FFT Теорема')
    plt.title("Результат Корреляции (Временная область)")
    plt.legend(); plt.grid(True)
    
    # Подготовка оси частот для результата свертки (он длиннее исходных сигналов!)
    N_conv = len(conv_fft)
    freqs_c = np.fft.fftfreq(N_conv, 1/fs)
    mask_c = freqs_c >= 0
    
    # 3. Амплитудный спектр СВЕРТКИ (Задание из методички!)
    plt.subplot(3, 2, 3)
    plt.stem(freqs_c[mask_c], np.abs(spec_conv)[mask_c], linefmt='r-', markerfmt='ro')
    plt.title("Амплитудный спектр Свертки")
    plt.xlim(0, 1500); plt.grid(True)
    
    # 4. Фазовый спектр СВЕРТКИ
    plt.subplot(3, 2, 4)
    plt.stem(freqs_c[mask_c], clean_phase(spec_conv)[mask_c], linefmt='r-', markerfmt='ro')
    plt.title("Фазовый спектр Свертки")
    plt.xlim(0, 1500); plt.grid(True)
    
    # 5. Проверка обратного преобразования (для свертки)
    # Восстанавливаем сигнал свертки из его спектра
    restored_conv = np.real(np.fft.ifft(spec_conv))
    
    plt.subplot(3, 1, 3)
    plt.plot(conv_fft, 'b', lw=3, alpha=0.5, label='Исходная свертка')
    plt.plot(restored_conv, 'k:', label='Восстановленная из спектра (IDFT)')
    plt.title("Проверка: Обратное преобразование Фурье от спектра свертки")
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig('page2_ops.png')

if __name__ == "__main__":
    run_lab()