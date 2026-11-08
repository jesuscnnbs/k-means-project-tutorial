# Contexto del proyecto — Agrupación de casas de California

## Objetivo

Clasificar casas de California según la región en la que se encuentran y la
renta media de su bloque censal. El resultado se usa para etiquetar datos no
supervisados con K-Means y, después, entrenar un clasificador supervisado que
asigne esas etiquetas a puntos nuevos.

## Dataset

`data/raw/housing.csv` — California Housing (censo de 1990). 20.640 filas, una
por grupo de bloques censales. Solo se usan tres columnas:

| Columna     | Significado                                             |
| ----------- | ------------------------------------------------------- |
| `Latitude`  | Latitud del bloque censal                               |
| `Longitude` | Longitud del bloque censal                              |
| `MedInc`    | Renta mediana del bloque, en decenas de miles de USD    |

Fuente: <https://breathecode.herokuapp.com/asset/internal-link?id=809&path=housing.csv>

## Decisiones de diseño

- **Split 80/20** con `random_state=42`. El test no participa en el ajuste del
  K-Means; solo se usa para comprobar la asignación de clusters y las métricas
  del clasificador.
- **`StandardScaler` antes de K-Means.** Sin escalar, la distancia euclídea
  estaría dominada por latitud y longitud (decenas de grados) y `MedInc`
  (0,5–15) sería irrelevante. Escalar da el mismo peso a las tres variables.
- **k = 6** fijado por el enunciado. En el notebook se revisan codo y
  silhouette solo como comprobación, no para cambiarlo.
- **Dos artefactos autocontenidos**: cada `.joblib` incluye su propio
  `StandardScaler`, así que se pueden usar sin recordar cómo se escaló.
- **El clasificador aprende etiquetas derivadas de las mismas features.** Un
  accuracy muy alto es esperado (los clusters de K-Means son regiones de
  Voronoi); su utilidad es asignar etiquetas sin cargar el K-Means.

## Estructura

```
data/raw/housing.csv                     dataset original
data/processed/housing_clusters.csv      train + test con columna cluster
models/kmeans_housing.joblib             StandardScaler + KMeans
models/classifier_housing.joblib         StandardScaler + clasificador
models/metadata.json                     features, métricas, perfiles, versiones
src/explore.ipynb                        análisis paso a paso
src/app.py                               pipeline reproducible de extremo a extremo
```

## Ejecución

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/app.py
jupyter lab src/explore.ipynb
```
