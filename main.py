import numpy as np
from src.generate_signals import generate_synthetic_signals
from src.preprocessing import preprocess_signal
from src.feature_extraction import extract_features
from src.feature_selection import select_features
from src.classification import classify

# Generate synthetic data
X_raw, y = generate_synthetic_signals()

# Preprocess and extract features
X_features = [extract_features(preprocess_signal(x)) for x in X_raw]

# Feature selection
X_selected = select_features(np.array(X_features), y, k=5)

# Classification
acc = classify(X_selected, y)
print(f"Classification Accuracy: {acc*100:.2f}%")
