import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.generate_signals import generate_synthetic_signals

X, y = generate_synthetic_signals()
np.save('data/synthetic_signals.npy', {'X': X, 'y': y})
print("Synthetic signals saved to data/synthetic_signals.npy")

data = np.load('data/synthetic_signals.npy', allow_pickle=True).item()
X = data['X']
y = data['y']
