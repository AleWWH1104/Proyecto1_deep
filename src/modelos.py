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