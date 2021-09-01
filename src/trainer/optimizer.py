from functools import partial

from classifiers import run_model
from skopt import gp_minimize, space


def optimize(X: pd.DataFrame, y: pd.DataFrame, params=Dict[Any, Any], n_splits: int = 5):

    _, valid_kp
