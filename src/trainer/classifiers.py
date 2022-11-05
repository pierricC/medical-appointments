"""Functions for training and evaluating classifiers."""

from typing import Any, Callable, Dict, Tuple

import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm


def train_classifier_lazy(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    custom_metric: Callable[..., Any] = matthews_corrcoef,
    ignore_warnings: bool = True,
    verbose: int = 0,
    predictions: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Pipeline, Dict[str, Pipeline]]:
    """Train and test different classifiers from sklearn API."""
    clf = LazyClassifier(
        verbose=verbose,
        ignore_warnings=ignore_warnings,
        custom_metric=custom_metric,
        predictions=predictions,
    )

    kpis, predictions = clf.fit(X_train, X_test, y_train, y_test)
    kpis = kpis.sort_values(custom_metric.__name__, ascending=False)
    best_model = clf.models[kpis.index[0]]
    return kpis, predictions, best_model, clf.models


def run_model(
    X: pd.DataFrame,
    y: pd.DataFrame,
    params: Dict[Any, Any],
    n_splits: int = 5,
    random_state: int = 12,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Stratified Cross-validation for rf classifier.

    Parameters
    ----------
    X
        Full dataset without labels
    y
        Full label
    params
        Classifier parameters in a dict
    n_splits
        Number of splits for the cross-validation
    """
    train_kpis = {}
    valid_kpis = {"valid_auc_mean": 0.0, "valid_mcc_mean": 0.0}
    i = 0

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for id_train, id_valid in tqdm(folds.split(X=X, y=y)):
        X_train, y_train = X.iloc[id_train], y.iloc[id_train]
        X_valid, y_valid = X.iloc[id_valid], y.iloc[id_valid]

        clf = RandomForestClassifier(**params)

        clf.fit(X_train, y_train)

        preds_train = clf.predict(X_train)
        preds_valid = clf.predict(X_valid)

        train_kpis[f"train_auc_fold_{i}"] = roc_auc_score(y_train, preds_train)
        train_kpis[f"train_mcc_fold_{i}"] = matthews_corrcoef(y_train, preds_train)

        valid_kpis[f"valid_auc_fold_{i}"] = roc_auc_score(y_valid, preds_valid)
        valid_kpis[f"valid_mcc_fold_{i}"] = matthews_corrcoef(y_valid, preds_valid)

        valid_kpis["valid_auc_mean"] += roc_auc_score(y_valid, preds_valid)
        valid_kpis["valid_mcc_mean"] += matthews_corrcoef(y_valid, preds_valid)

        i += 1

    valid_kpis["valid_auc_mean"] /= n_splits
    valid_kpis["valid_mcc_mean"] /= n_splits

    return train_kpis, valid_kpis
