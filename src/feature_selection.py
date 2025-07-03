from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
import numpy as np

def select_features(X, y, k=10):
    # You can swap with PCA if needed
    scores = mutual_info_classif(X, y)
    idx = np.argsort(scores)[-k:]
    return X[:, idx]
