"""Utilities for plotting."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def plot_features_importances(X: pd.DataFrame, y: pd.Series) -> None:
    """Plot the feature importance using a RandomForestclf."""
    model = RandomForestClassifier()
    model.fit(X, y)

    importances = model.feature_importances_
    idxs = np.argsort(importances)
    plt.title("Feature Importance")
    plt.barh(range(len(idxs)), importances[idxs], align="center")
    plt.yticks(range(len(idxs)), [X.columns[i] for i in idxs])
    plt.xlabel("Random Forest Feature Importance")
    plt.show()
