import numpy as np
from src.generate_signals import generate_synthetic_signals
from src.preprocessing import preprocess_signal
from src.feature_extraction import extract_features
from src.feature_selection import select_features
from src.classification import classify
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ---------- Classify Function ----------
def classify(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = SVC(kernel='rbf', C=1, gamma='scale')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc, clf, X_test, y_test


# ---------- Graph: Raw vs Preprocessed Signal ----------
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

# ---------- Main Classification Pipeline ----------
# Generate synthetic data
X_raw, y = generate_synthetic_signals()

# Preprocess and extract features
X_features = [extract_features(preprocess_signal(x)) for x in X_raw]

# Feature selection
X_selected = select_features(np.array(X_features), y, k=5)

# Classification (updated to return extra values)
acc, clf, X_test, y_test = classify(X_selected, y)

# ---------- Confusion Matrix ----------
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='viridis')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ---------- Accuracy ----------
print(f"Classification Accuracy: {acc*100:.2f}%")
