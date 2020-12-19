from scipy.io.wavfile import read
from scipy.signal.windows import blackman, hamming
from scipy import fft

import numpy as np
import matplotlib.pyplot as plt
import os

def print_windows():
    windows = ['blackman', 'hamming', 'square']
    count = 0
    for window in windows:
        print("{} - {}".format(count,window))
        count += 1

def print_files_indir():
    # Simple function to print the wav files in the current dir
    files = sorted(os.listdir())
    count = 0
    videos = []

    for file in files:
        if file.endswith('.wav'):
            print("{} - {}".format(count,file))
            videos.append(file)
            count += 1
    return count, videos


def print_powers_of_2():
    ns = []
    for i in range(8):
        num = np.power(2,i+8)
        print("{} - {}".format(i,num))
        ns.append(num)
    return ns

def compute_magnitude_windowed_fft(signal, window):
    if signal.dtype == 'uint8':
        signal = np.array(signal, dtype=np.int32) - 128
        signal = signal / 128
    if signal.dtype == 'int16':
        signal = signal / 32768
    w_windowed = signal * window
    x_mag = abs(fft(w_windowed))
    dB_mag = 10*np.log(x_mag)
    return dB_mag


def compute_spectrogram(signal, sr, fft_size, hop_size, window):
    Mag_size = fft_size//2
    spectrogam = []
    idx_last = fft_size
    while idx_last < len(signal):
        current_slice = signal[idx_last - fft_size : idx_last]
        spectrum = compute_magnitude_windowed_fft(current_slice, blackman(fft_size))[Mag_size:-1]
        spectrogam.append(spectrum)
        idx_last += hop_size

    x_axis = np.linspace(0, sr//2, Mag_size)
    return np.array(spectrogam)

def plot_spectrogram(spectrogram, sr, n_samples, filename, fft_size, hop_size):
    plt.clf()
    plt.imshow(spectrogram.T, extent=[0, n_samples//sr , 20, sr//2], aspect = 'auto')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.yscale('symlog')
    plt.title('Spectrogram')
    plt.colorbar().set_label('Amplitude [dB]')
    plt.savefig('{}_{}_{}.png'.format(filename.split('.')[0],fft_size,hop_size))

menu = True
while menu:
    _, videos = print_files_indir()
    idx = int(input("Choose tha audio to compute the Spectrogram"))
    filename = videos[idx]

    ns = print_powers_of_2()
    idx = int(input("Choose the fft_size"))
    fft_size = ns[idx]

    print_powers_of_2()
    idx = int(input("Choose the fft_size"))
    hop_size = ns[idx]

    windows = [blackman(fft_size), hamming(fft_size), np.ones(fft_size)]
    print_windows()
    idx = int(input("Choose the window type"))
    window = windows[idx]
    sr, signal = read(filename)

    spec = compute_spectrogram(signal, sr, fft_size, hop_size, window)
    plot_spectrogram(spec, sr, len(signal), filename, fft_size, hop_size)
