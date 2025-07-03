from scipy.signal import butter, filtfilt
import numpy as np

def bandpass_filter(signal, lowcut=400, highcut=12000, fs=50000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

def preprocess_signal(signal):
    filtered = bandpass_filter(signal)
    smoothed = np.convolve(filtered, np.ones(5)/5, mode='same')
    normalized = (smoothed - np.mean(smoothed)) / np.max(np.abs(smoothed))
    return normalized
