"""Set of functions to preprocess the data before modelling."""
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder


def get_date(serie: pd.Series, separator: str) -> pd.Series:
    """Transform a serie into a datetime series."""
    times = []

    for i, date in enumerate(serie):
        partition = date.partition(separator)
        period = partition[0]
        time = partition[2].replace("Z", "")

        times.append(period + " " + time)

    return pd.to_datetime(times)


def num_to_cat(df: pd.DataFrame, max_unique_values: int = 10, plot: bool = True) -> pd.DataFrame:
    """Convert numerical features with few values to categorical."""
    df_cp = df.copy()
    num_candidates = list(df_cp.dtypes[df_cp.dtypes != "object"].index.values)
    unique_counts = df_cp.loc[:, num_candidates].nunique().sort_values()

    if plot:
        plt.figure(figsize=(20, 5))
        sns.barplot(unique_counts.index, unique_counts.values, palette="Oranges_r")
        plt.xticks(rotation=90)
        plt.yscale("log")
        plt.show()

    num_to_cats_ind = np.where(unique_counts < max_unique_values)
    num_to_cats = unique_counts.iloc[num_to_cats_ind].index
    for feat in num_to_cats:
        df_cp[feat] = df_cp[feat].astype("object")

    return df_cp


def feature_engineering(df: pd.DataFrame):

    data = df.copy()
    times_sch = get_date(data["ScheduledDay"], "T")
    times_apt = get_date(data["AppointmentDay"], "T")

    data["AppointmentDay"] = times_apt
    data["ScheduledDay"] = times_sch

    data["month_sch"] = times_sch.month
    data["day_sch"] = times_sch.day
    data["hour_sch"] = times_sch.hour
    data["minute_sch"] = times_sch.minute
    data["second_sch"] = times_sch.second

    data["month_apt"] = times_apt.month
    data["day_apt"] = times_apt.day

    data = data.drop(["ScheduledDay", "AppointmentDay"], axis=1)

    return data


def labelencode(df: pd.DataFrame, col_to_encode: List[str]) -> pd.DataFrame:
    """Labelencode each categorical variables in the dataframe."""
    df_cp = df.copy()
    for col in col_to_encode:
        # initialize labelencoder for each categorical column
        encoder = LabelEncoder()

        # fit label encoder on all data
        encoder.fit(df_cp[col])

        # transform all the data
        df_cp.loc[:, col] = encoder.transform(df_cp[col])

    return df_cp


# def categorical_feat_engineering(
#     df: pd.DataFrame, cat_cols: List[str], combination_size: int
# ) -> pd.DataFrame:
#     """Create all combinations of the categorical columns provided wi."""
