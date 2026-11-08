# Agrupación de casas de California con K-Means

Clasificación de casas por **región** y **renta media** usando el dataset
*California Housing* (censo de 1990, 20.640 grupos de bloques censales). El
proyecto encadena un modelo no supervisado y otro supervisado:

1. **K-Means (k = 6)** agrupa las casas usando `Latitude`, `Longitude` y
   `MedInc`.
2. Las etiquetas generadas se añaden al dataset como columna `cluster`.
3. Un **clasificador supervisado** aprende a reproducir esas etiquetas para
   puntos nuevos.
4. Ambos modelos se guardan en `models/`.

## Resultados

**K-Means (k = 6, train escalado).** Inercia 8.069,2 y silhouette de test 0,399.
Los grupos combinan región y renta:

| Cluster | % train | MedInc medio | Zona aproximada |
| :-----: | :-----: | :----------: | --------------- |
| 0 | 8,0 % | 2,6 | Valle Central (Fresno–Bakersfield) |
| 1 | 20,5 % | 5,3 | Los Ángeles, Orange County y San Diego |
| 2 | 2,7 % | 10,4 | Enclaves de renta alta en todo el estado |
| 3 | 23,8 % | 2,8 | Sacramento, norte y costa |
| 4 | 13,6 % | 5,5 | Bahía de San Francisco y norte |
| 5 | 31,4 % | 2,8 | Inland Empire y sur metropolitano |

**Clasificador supervisado.** Random Forest gana la validación cruzada
(0,9939 ± 0,0008) frente a Logistic Regression (0,9901) y KNN (0,9882). En
test: accuracy **0,9956**, balanced accuracy **0,9923** y F1 macro **0,9940**
(18 errores de 4.128 puntos). El accuracy es alto porque las etiquetas se
derivan de las mismas features, no porque el cluster sea un objetivo externo.

| Artefacto | Contenido |
| --------- | --------- |
| `models/kmeans_housing.joblib` | `StandardScaler` + `KMeans(n_clusters=6)` |
| `models/classifier_housing.joblib` | `StandardScaler` + `RandomForestClassifier` |
| `models/metadata.json` | features, métricas, perfiles de cluster y versiones |
| `data/processed/housing_clusters.csv` | train + test etiquetados con `cluster` |

El detalle de métricas, gráficos y conclusiones está en
[`src/explore.ipynb`](src/explore.ipynb). El resumen y las decisiones de diseño
están en [`CONTEXT.md`](CONTEXT.md).

## Estructura

```
data/raw/housing.csv                     dataset original
data/processed/housing_clusters.csv      dataset etiquetado (train + test)
models/                                  artefactos .joblib y metadata.json
src/explore.ipynb                        exploración, clustering y evaluación
src/app.py                               pipeline reproducible de extremo a extremo
```

## Instalación y ejecución

Requiere Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Reproducir el pipeline completo y regenerar los artefactos:

```bash
python src/app.py
```

Abrir el análisis:

```bash
jupyter lab src/explore.ipynb
```

## Uso de los modelos

```python
import joblib
import pandas as pd

kmeans = joblib.load("models/kmeans_housing.joblib")
classifier = joblib.load("models/classifier_housing.joblib")

nuevas_casas = pd.DataFrame(
    {
        "Latitude": [34.05, 37.77],
        "Longitude": [-118.24, -122.42],
        "MedInc": [3.5, 8.1],
    }
)

print(kmeans.predict(nuevas_casas))     # cluster directo de K-Means
print(classifier.predict(nuevas_casas)) # mismo cluster vía clasificador
```

## Nota metodológica

Las etiquetas del clasificador provienen del propio K-Means, calculado sobre
las mismas tres features. Por eso un accuracy cercano al 100% es esperado y no
demuestra capacidad predictiva sobre una variable externa: demuestra que el
clasificador reproduce fielmente la partición del K-Means.
