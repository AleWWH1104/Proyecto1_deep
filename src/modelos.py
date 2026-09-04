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


def _entrenar_bucle(modelo, secuencias_train, etiquetas_train, secuencias_val, etiquetas_val,
                     n_epocas=10, batch_size=256, lr=1e-3):
    dataset_train = SecuenciaDataset(secuencias_train, etiquetas_train)
    dataset_val = SecuenciaDataset(secuencias_val, etiquetas_val)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

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


def entrenar_modelo_B(secuencias_train, etiquetas_train, secuencias_val, etiquetas_val,
                       n_features, hidden_size=32, n_epocas=10, batch_size=256,
                       lr=1e-3, semilla=42):
    torch.manual_seed(semilla)
    modelo = ModeloLSTM(n_features, hidden_size)
    return _entrenar_bucle(modelo, secuencias_train, etiquetas_train, secuencias_val, etiquetas_val,
                            n_epocas=n_epocas, batch_size=batch_size, lr=lr)


def predecir_modelo_B(modelo, secuencias, batch_size=256):
    modelo.eval()
    X = torch.tensor(secuencias, dtype=torch.float32)
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            salida = modelo(X[i:i + batch_size])
            scores.append(salida.numpy())
    return np.concatenate(scores)


class ModeloLSTM_Embedding(nn.Module):
    """LSTM que reemplaza el código numérico de categoría por un embedding aprendido.

    Asume que la columna de categoría es la de índice `indice_categoria` dentro
    de las features de evento (columnas_evento = [amt, category_cod, hora_del_dia,
    dias_desde_ultima_tx] en datos.construir_secuencias, por lo que el índice por
    defecto es 1).
    """

    def __init__(self, n_features, n_categorias, dim_embedding=4, hidden_size=32,
                 indice_categoria=1):
        super().__init__()
        self.indice_categoria = indice_categoria
        self.embedding = nn.Embedding(n_categorias, dim_embedding)
        n_features_continuas = n_features - 1
        self.lstm = nn.LSTM(n_features_continuas + dim_embedding, hidden_size, batch_first=True)
        self.salida = nn.Linear(hidden_size, 1)

    def forward(self, x):
        idx = self.indice_categoria
        codigo = x[:, :, idx].long().clamp(min=0)
        emb = self.embedding(codigo)
        continuas = torch.cat([x[:, :, :idx], x[:, :, idx + 1:]], dim=2)
        entrada = torch.cat([continuas, emb], dim=2)
        _, (h_n, _) = self.lstm(entrada)
        ultimo_estado = h_n[-1]
        logit = self.salida(ultimo_estado)
        return torch.sigmoid(logit).squeeze(1)


def entrenar_modelo_C(secuencias_train, etiquetas_train, secuencias_val, etiquetas_val,
                       n_features, n_categorias, dim_embedding=4, hidden_size=32,
                       n_epocas=10, batch_size=256, lr=1e-3, semilla=42):
    torch.manual_seed(semilla)
    modelo = ModeloLSTM_Embedding(n_features, n_categorias, dim_embedding, hidden_size)
    return _entrenar_bucle(modelo, secuencias_train, etiquetas_train, secuencias_val, etiquetas_val,
                            n_epocas=n_epocas, batch_size=batch_size, lr=lr)


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


def analisis_costo(y_true, scores, costo_fraude_no_detectado=4200.0, costo_bloqueo_legitimo=180.0,
                    umbrales=None):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if umbrales is None:
        umbrales = np.linspace(0.01, 0.99, 197)

    detalle = []
    for u in umbrales:
        y_pred = (scores >= u).astype(int)
        fraudes_no_detectados = int(np.sum((y_pred == 0) & (y_true == 1)))
        bloqueos_legitimos = int(np.sum((y_pred == 1) & (y_true == 0)))
        costo = (fraudes_no_detectados * costo_fraude_no_detectado
                 + bloqueos_legitimos * costo_bloqueo_legitimo)
        detalle.append({
            "umbral": float(u),
            "fraudes_no_detectados": fraudes_no_detectados,
            "bloqueos_legitimos": bloqueos_legitimos,
            "costo": costo,
        })

    mejor = min(detalle, key=lambda r: r["costo"])
    return mejor["umbral"], detalle