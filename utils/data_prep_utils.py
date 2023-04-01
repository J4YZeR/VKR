import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def drop_outliers_iqr(df, iqr_bound=(0.25, 0.75), IQR_k=7):
    Q1 = df.quantile(iqr_bound[0])
    Q3 = df.quantile(iqr_bound[1])
    IQR = Q3 - Q1
    return df[~((df < (Q1 - IQR_k * IQR)) | (df > (Q3 + IQR_k * IQR))).any(axis=1)]
def visualise_data(df: pd.DataFrame, remove_outliers=True, IQR_k=7):
    df_to_visualize = df.copy()
    if remove_outliers:
        df_to_visualize = drop_outliers_iqr(df_to_visualize, IQR_k=IQR_k)
    df_to_visualize.hist(bins=50, figsize=(20, 15))
    plt.figure()
    fig, axes = plt.subplots(5, int(np.ceil(len(df_to_visualize.columns) / 5)), figsize=(20, 15))
    for i,el in enumerate(list(df_to_visualize.columns.values)):
        a = df_to_visualize.boxplot([el], ax=axes.flatten()[i])
    plt.tight_layout()
    plt.show()