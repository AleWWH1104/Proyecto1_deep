import pandas as pd
import numpy as np


def cargar_datos(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
    return df


def split_temporal(df: pd.DataFrame, frac_train=0.6, frac_val=0.2,
                    col_tiempo="trans_date_trans_time"):
    df = df.sort_values(col_tiempo)
    n = len(df)
    t1 = df[col_tiempo].iloc[int(n * frac_train)]
    t2 = df[col_tiempo].iloc[int(n * (frac_train + frac_val))]

    train = df[df[col_tiempo] < t1].copy()
    val = df[(df[col_tiempo] >= t1) & (df[col_tiempo] < t2)].copy()
    test = df[df[col_tiempo] >= t2].copy()

    print(f"train: {len(train)} ({train[col_tiempo].min()} -> {train[col_tiempo].max()})")
    print(f"val:   {len(val)} ({val[col_tiempo].min()} -> {val[col_tiempo].max()})")
    print(f"test:  {len(test)} ({test[col_tiempo].min()} -> {test[col_tiempo].max()})")
    return train, val, test
