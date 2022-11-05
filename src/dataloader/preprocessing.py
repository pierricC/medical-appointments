"""Set of functions to preprocess the data before modelling."""
from collections import Counter
from typing import List, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder


def num_to_cat(
    df: pd.DataFrame, max_unique_values: int = 10, plot: bool = True
) -> pd.DataFrame:
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


def get_date(serie: pd.Series, separator: str) -> pd.Series:
    """Transform a serie into a datetime series."""
    times = []

    for i, date in enumerate(serie):
        partition = date.partition(separator)
        period = partition[0]
        time = partition[2].replace("Z", "")

        times.append(period + " " + time)

    return pd.to_datetime(times)


def time_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Get time features from 'ScheduledDay' and 'AppointmentDay'."""
    data = df.copy()
    times_sch = get_date(data["ScheduledDay"], "T")
    times_apt = get_date(data["AppointmentDay"], "T")

    data["AppointmentDay"] = times_apt
    data["ScheduledDay"] = times_sch

    data["Month_sch"] = times_sch.month
    data["Day_sch"] = times_sch.day
    data["Hour_sch"] = times_sch.hour
    data["Minute_sch"] = times_sch.minute
    data["Second_sch"] = times_sch.second

    data["Month_apt"] = times_apt.month
    data["Day_apt"] = times_apt.day

    data = data.drop(["ScheduledDay", "AppointmentDay"], axis=1)

    return data


def get_total_occurrence(
    df: pd.DataFrame, feature: str = "PatientID", max_frequency: int = 3
) -> pd.DataFrame:
    """
    Get the frequency of each value in the feature column.

    Create a new categorical feature out of it with a number max of categories.
    If a value appears more than max frequency, then its frequency is equal to max_frequency.
    """
    occurrence_dict = dict(Counter(df[feature])).items()  # type: ignore
    column_name = f"Nb_occurrence_{feature}"
    df_occurrence = pd.DataFrame(data=occurrence_dict, columns=[feature, column_name])
    df_occurrence.loc[
        df_occurrence[column_name].value_counts()[df_occurrence[column_name]].index
        >= max_frequency,
        column_name,
    ] = max_frequency

    df = pd.merge(df, df_occurrence, on=feature, how="outer")

    # we drop the feature since we don't need it anymore
    df = df.drop(feature, axis=1)
    return df


def feature_to_bin(
    df: pd.DataFrame, feature_name: str, bins: Union[int, List[int]]
) -> pd.DataFrame:
    """Create a categorical feature from a numerical by putting values into bins."""
    df[f"{feature_name}_bins"] = pd.cut(df[feature_name], bins=bins, labels=False)

    return df


def labelencode(df: pd.DataFrame, col_to_encode: List[str]) -> pd.DataFrame:
    """Labelencode each categorical variables in the dataframe."""
    df_cp = df.copy()
    for col in col_to_encode:
        # initialize labelencoder for each categorical column
        encoder = LabelEncoder()

        # fit & transform all the data
        df_cp.loc[:, col] = encoder.fit_transform(df_cp[col])

    return df_cp
