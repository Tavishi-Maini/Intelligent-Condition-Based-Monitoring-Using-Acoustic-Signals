# 🛠 Intelligent Condition-Based Monitoring using Acoustic Signals

This project implements a simplified version of the fault detection pipeline described in the research paper:  
**"Intelligent Condition-Based Monitoring Using Acoustic Signals for Air Compressors"**  
It uses synthetic acoustic signals to simulate healthy and faulty compressor states, followed by preprocessing, feature extraction, feature selection, and classification using machine learning.

---

## 📁 Project Structure

acoustic_fault_detection/
│
├── data/
│ └── synthetic_signals.npy # Saved synthetic data (X, y)
│ └── synthetic_signals.py # Script to generate and save synthetic data
│
├── src/
│ ├── generate_signals.py # Signal generation (healthy/faulty)
│ ├── preprocessing.py # Filtering, clipping, smoothing, normalization
│ ├── feature_extraction.py # Time and frequency domain features
│ ├── feature_selection.py # Mutual Information-based feature selection
│ └── classification.py # SVM-based classifier
│
└── main.py # Runs the full pipeline


## How the Pipeline Works

1. **Signal Generation**
   - Synthetic acoustic signals are generated using sine + noise components.
   - Two classes: healthy (label `0`) and faulty (label `1`).

2. **Preprocessing**
   - Bandpass filter (400 Hz – 12 kHz)
   - Clip 1-second stable segment
   - Smooth using moving average
   - Normalize to range [–1, 1]

3. **Feature Extraction**
   - Time domain features: RMS, skewness, etc.
   - Frequency domain features: FFT-based energy

4. **Feature Selection**
   - Mutual Information method selects most relevant features

5. **Classification**
   - Uses Support Vector Machine (SVM) with RBF kernel
   - Prints accuracy on held-out test set

## 📦 Requirements

Install dependencies using pip:

```bash
pip install numpy scipy scikit-learn matplotlib

##▶️ How to Run

##First, generate and save the synthetic data:
python data/synthetic_signals.py
##Then run the main pipeline:
python main.py


