import numpy as np
from src.generate_signals import generate_synthetic_signals
from src.preprocessing import preprocess_signal
from src.feature_extraction import extract_features
from src.feature_selection import select_features
from src.classification import classify
import matplotlib.pyplot as plt

#graph
X, _ = generate_synthetic_signals(n_samples=2)
raw = X[0]
preprocessed = preprocess_signal(raw)

plt.figure(figsize=(12, 4))
plt.plot(raw, label='Raw Signal', alpha=0.6)
plt.plot(preprocessed, label='Preprocessed Signal', alpha=0.8)
plt.title("Raw vs Preprocessed Acoustic Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Generate synthetic data
X_raw, y = generate_synthetic_signals()

# Preprocess and extract features
X_features = [extract_features(preprocess_signal(x)) for x in X_raw]

# Feature selection
X_selected = select_features(np.array(X_features), y, k=5)

# Classification
acc = classify(X_selected, y)
print(f"Classification Accuracy: {acc*100:.2f}%")

