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