import numpy as np
from scipy.stats import skew, kurtosis

def extract_features(signal, fs=50000):
    # Time domain
    features = [
        np.mean(np.abs(signal)), np.std(signal),
        skew(signal), kurtosis(signal),
        np.max(signal), np.min(signal)
    ]

    # Frequency domain
    fft_vals = np.abs(np.fft.rfft(signal))
    fft_freqs = np.fft.rfftfreq(len(signal), 1/fs)
    fft_energy = np.sum(fft_vals**2)
    features.append(fft_energy)
    return features
