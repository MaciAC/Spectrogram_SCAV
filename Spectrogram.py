from scipy.io.wavfile import read
from scipy.signal.windows import blackman, hamming
from scipy import fft

import numpy as np
import matplotlib.pyplot as plt


def compute_magnitude_windowed_fft(signal, window):
    signal = signal / np.max(np.abs(signal))
    w_windowed = signal * window
    x_mag = abs(fft(w_windowed))
    dB_mag = 10*np.log(x_mag)
    dB_mag[dB_mag < 0] = 0
    return dB_mag


def compute_spectrogram(signal, sr, fft_size, hop_size, window):
    spectrogam = []
    Mag_size = fft_size//2-1

    idx_last = fft_size
    while idx_last < len(signal):
        current_slice = signal[idx_last - fft_size : idx_last]
        spectrum = compute_magnitude_windowed_fft(current_slice, blackman(fft_size))[0:Mag_size]
        spectrogam.append(spectrum)
        idx_last += hop_size

    x_axis = np.linspace(0, sr//2, Mag_size)
    return spectrogam

def plot_spectrogram(spectrogram):
    
sr, signal = read("xirp.wav")

compute_spectrogram(signal, sr, 1024, 128, "s")
