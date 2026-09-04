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


def construir_agregados(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).copy()
    df = df.set_index("trans_date_trans_time")

    agregados = []
    for cc_num, grupo in df.groupby("cc_num"):
        g = grupo.copy()
        g["monto_prom_24h"] = g["amt"].rolling("24h").mean()
        g["n_tx_ultima_hora"] = g["amt"].rolling("1h").count()
        g["monto_max_dia"] = g["amt"].rolling("1D").max()
        g["diversidad_comercio"] = (
            g["category_cod"].rolling("24h").apply(lambda x: pd.Series(x).nunique(), raw=True)
        )
        agregados.append(g)

    return pd.concat(agregados).reset_index()


def construir_secuencias(df: pd.DataFrame, longitud=10):
    columnas_evento = ["amt", "category_cod", "hora_del_dia", "dias_desde_ultima_tx"]
    secuencias, etiquetas, ids = [], [], []

    for cc_num, grupo in df.groupby("cc_num"):
        g = grupo.sort_values("trans_date_trans_time").reset_index(drop=True)
        valores = g[columnas_evento].values

        for i in range(len(g)):
            inicio = max(0, i - longitud + 1)
            ventana = valores[inicio:i + 1]
            if len(ventana) < longitud:
                relleno = np.zeros((longitud - len(ventana), len(columnas_evento)))
                ventana = np.vstack([relleno, ventana])
            secuencias.append(ventana)
            etiquetas.append(g.loc[i, "is_fraud"])
            ids.append(g.loc[i, "trans_num"])

    return np.array(secuencias), np.array(etiquetas), ids


def preparar_features_evento(df: pd.DataFrame, cat2cod: dict = None):
    df = df.copy()
    if cat2cod is None:
        categorias = sorted(df["category"].unique())
        cat2cod = {c: i for i, c in enumerate(categorias)}

    df["category_cod"] = df["category"].map(cat2cod).fillna(-1)
    df["hora_del_dia"] = df["trans_date_trans_time"].dt.hour
    df["dias_desde_ultima_tx"] = (
        df.groupby("cc_num")["trans_date_trans_time"].diff().dt.total_seconds() / 86400
    ).fillna(0)

    return df, cat2cod
