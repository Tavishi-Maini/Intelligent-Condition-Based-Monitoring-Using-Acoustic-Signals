import numpy as np

def generate_synthetic_signals(n_samples=200, length=2500, fs=50000):
    """
    Generate synthetic acoustic signals: 100 healthy, 100 faulty.
    """
    t = np.linspace(0, length/fs, length)
    healthy = [np.sin(2 * np.pi * 1000 * t) + 0.05*np.random.randn(length) for _ in range(n_samples // 2)]
    faulty = [np.sin(2 * np.pi * 1000 * t) + 0.5*np.sin(2 * np.pi * 2000 * t) + 0.05*np.random.randn(length) for _ in range(n_samples // 2)]
    X = np.array(healthy + faulty)
    y = np.array([0]*(n_samples//2) + [1]*(n_samples//2))  # 0 = healthy, 1 = faulty
    return X, y
