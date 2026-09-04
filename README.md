# Proyecto 1 — El orden revela: monitoreo transaccional

Detección de fraude con tarjeta comparando un modelo sobre variables agregadas (A) contra un modelo secuencial LSTM (B), más una apuesta propia con embedding de categoría (C).

## Reproducción

1. Instalar `uv` si no lo tienen: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Descargar `fraudTrain.csv` del dataset Sparkov en [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection) y colocarlo en `data/fraudTrain.csv`. Es el único CSV que usa el notebook; la partición train/val/test se hace por tiempo dentro del mismo archivo.
3. Instalar dependencias:
    ```bash
    uv sync
    ```
4. Abrir y correr el notebook de arriba a abajo:
    ```bash
    uv run jupyter notebook proyecto_ayala_quezada.ipynb
    ```
5. Al final del notebook se guardan los artefactos en `artefactos/`.

## Versiones

Python ≥3.9, gestionado con `uv`. Dependencias principales: `torch`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `joblib` (ver `pyproject.toml` para versiones exactas).

## Uso de IA

Usamos Claude Code como asistente durante el proyecto: para depurar código, redactar el README, y para armar los artefactos finales.

Tres decisiones técnicas donde se usó IA como apoyo:

1. **Longitud de secuencia (10 transacciones).** Alternativas consideradas: 5 (menos contexto, menos padding) y 20 (más contexto, más tarjetas sin historia suficiente). Nos quedamos con 10 como punto intermedio razonable; no lo comparamos empíricamente contra 5 y 20, así que es una decisión de diseño, no una elegida por resultado.
2. **Umbral de bloqueo por costo (0.06) en vez de 0.5 o el umbral que maximiza F1.** La evidencia fue la curva de costo esperado (Figura 1 del informe): con Q4,200 por fraude no detectado contra Q180 por bloqueo de más, el mínimo cae en 0.06.
3. **Embedding de categoría en el modelo C en vez de one-hot o dejar el código numérico crudo (como en B).** Se descartó one-hot por aumentar mucho el tamaño de la secuencia. La evidencia fue la mejora de +0.03 en AUC-PR de validación, fijada como umbral de éxito antes de correr el experimento, confirmada después en prueba.

## Candidato al Proyecto Final

- **Modelo que conservaríamos:** el Modelo C (LSTM con embedding de categoría). Artefacto en `artefactos/modelo_C.pt`, con los parámetros de preparación en `artefactos/preproc_params.json` (diccionario de categorías, largo de secuencia, columnas usadas, umbral operativo 0.06).
- **Quién usaría el puntaje:** el área de riesgos del banco, para decidir si bloquea, deja pasar o manda a revisión manual una transacción en tiempo real.
- **Contrato preliminar:** entrada = últimas 10 transacciones de la tarjeta (monto, categoría, hora del día, días desde la transacción anterior); salida = un puntaje de riesgo entre 0 y 1.
- **Límites y riesgos:** entrenado con datos simulados (Sparkov), no con transacciones reales del banco; exhaustividad más baja en categorías con pocos fraudes en entrenamiento (viajes, supermercados en línea, tiendas varias); no probado contra mecanismos de fraude nuevos. Para producción faltaría reentrenar con datos reales y definir cómo monitorear si el modelo se degrada con el tiempo.
