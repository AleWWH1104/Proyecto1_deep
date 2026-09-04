import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
)
 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
 
 
def entrenar_modelo_A(X_train, y_train):
    scaler = StandardScaler()
    X_train_esc = scaler.fit_transform(X_train)
 
    modelo = GradientBoostingClassifier(random_state=42)
    modelo.fit(X_train_esc, y_train)
    return modelo, scaler
 
 
def predecir_modelo_A(modelo, scaler, X):
    X_esc = scaler.transform(X)
    return modelo.predict_proba(X_esc)[:, 1]

class SecuenciaDataset(Dataset):
    def __init__(self, secuencias, etiquetas):
        self.X = torch.tensor(secuencias, dtype=torch.float32)
        self.y = torch.tensor(etiquetas, dtype=torch.float32)
 
    def __len__(self):
        return len(self.y)
 
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
 
 
class ModeloLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.salida = nn.Linear(hidden_size, 1)
 
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        ultimo_estado = h_n[-1]
        logit = self.salida(ultimo_estado)
        return torch.sigmoid(logit).squeeze(1)


def entrenar_modelo_B(secuencias_train, etiquetas_train, secuencias_val, etiquetas_val,
                       n_features, hidden_size=32, n_epocas=10, batch_size=256,
                       lr=1e-3, semilla=42):
    torch.manual_seed(semilla)

    dataset_train = SecuenciaDataset(secuencias_train, etiquetas_train)
    dataset_val = SecuenciaDataset(secuencias_val, etiquetas_val)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    modelo = ModeloLSTM(n_features, hidden_size)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    criterio = nn.BCELoss()

    for epoca in range(n_epocas):
        modelo.train()
        perdida_train = 0.0
        for X_batch, y_batch in loader_train:
            optimizador.zero_grad()
            salida = modelo(X_batch)
            perdida = criterio(salida, y_batch)
            perdida.backward()
            optimizador.step()
            perdida_train += perdida.item() * len(y_batch)
        perdida_train /= len(dataset_train)

        modelo.eval()
        perdida_val = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader_val:
                salida = modelo(X_batch)
                perdida_val += criterio(salida, y_batch).item() * len(y_batch)
        perdida_val /= len(dataset_val)

        print(f"Época {epoca + 1}/{n_epocas} - pérdida train: {perdida_train:.4f} - pérdida val: {perdida_val:.4f}")

    return modelo


def predecir_modelo_B(modelo, secuencias, batch_size=256):
    modelo.eval()
    X = torch.tensor(secuencias, dtype=torch.float32)
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            salida = modelo(X[i:i + batch_size])
            scores.append(salida.numpy())
    return np.concatenate(scores)


def evaluar(y_true, scores, umbral=0.5):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_pred = (scores >= umbral).astype(int)
    return {
        "auc_pr": average_precision_score(y_true, scores),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def prueba_permutacion(modelo, secuencias, etiquetas, semilla=42):
    rng = np.random.default_rng(semilla)

    scores_original = predecir_modelo_B(modelo, secuencias)
    auc_original = average_precision_score(etiquetas, scores_original)

    secuencias_barajadas = secuencias.copy()
    for i in range(len(secuencias_barajadas)):
        permutacion = rng.permutation(secuencias_barajadas.shape[1])
        secuencias_barajadas[i] = secuencias_barajadas[i][permutacion]

    scores_barajadas = predecir_modelo_B(modelo, secuencias_barajadas)
    auc_barajada = average_precision_score(etiquetas, scores_barajadas)

    return auc_original, auc_barajada


def analisis_costo(y_true, scores, costo_fp=1.0, costo_fn=10.0, umbrales=None):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if umbrales is None:
        umbrales = np.linspace(0.01, 0.99, 99)

    resultados = []
    for u in umbrales:
        y_pred = (scores >= u).astype(int)
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        costo = fp * costo_fp + fn * costo_fn
        resultados.append({"umbral": u, "fp": fp, "fn": fn, "costo": costo})

    mejor = min(resultados, key=lambda r: r["costo"])
    return {"resultados": resultados, "mejor_umbral": mejor["umbral"], "costo_minimo": mejor["costo"]}