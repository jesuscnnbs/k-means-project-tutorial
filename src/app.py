"""Agrupación de casas de California por región e ingreso medio.

Pipeline reproducible del proyecto:

1. Carga ``data/raw/housing.csv`` y usa solo ``Latitude``, ``Longitude`` y
   ``MedInc``.
2. Divide en train/test y escala las features.
3. Entrena un K-Means de 6 clusters con el conjunto de train.
4. Usa las etiquetas del K-Means para entrenar un clasificador supervisado.
5. Guarda ambos modelos en ``models/`` y el dataset etiquetado en
   ``data/processed/``.

Ejecución: ``python src/app.py`` desde la raíz del repositorio.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Anclamos las rutas a la raíz del repositorio para que funcione desde
# cualquier directorio de trabajo.
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "housing.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "housing_clusters.csv"
MODELS_DIR = ROOT / "models"
KMEANS_PATH = MODELS_DIR / "kmeans_housing.joblib"
CLASSIFIER_PATH = MODELS_DIR / "classifier_housing.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Columnas que pide el enunciado: ubicación y renta media del bloque censal.
FEATURES = ["Latitude", "Longitude", "MedInc"]
# 6 clusters fijados por el enunciado.
N_CLUSTERS = 6
# Semilla única para split, K-Means y modelos: resultados reproducibles.
RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPECTED_ROWS = 20_640


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Carga el CSV y valida el contrato mínimo del dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Descarga housing.csv en data/raw/."
        )

    df = pd.read_csv(path)

    missing = set(FEATURES).difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")

    # Trabajamos solo con las tres columnas del enunciado.
    df = df[FEATURES].copy()

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Se esperaban {EXPECTED_ROWS} filas, hay {len(df)}")
    if df.isna().any().any():
        raise ValueError("Hay valores nulos en las features seleccionadas")
    if df["MedInc"].le(0).any():
        raise ValueError("MedInc debe ser estrictamente positivo")
    if not df["Latitude"].between(32, 43).all():
        raise ValueError("Latitud fuera del rango de California")
    if not df["Longitude"].between(-125, -113).all():
        raise ValueError("Longitud fuera del rango de California")

    return df


def split_data(df: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """Devuelve los índices de train y test con la semilla del proyecto."""
    train_index, test_index = train_test_split(
        df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return train_index, test_index


def build_kmeans() -> Pipeline:
    """K-Means dentro de un pipeline con escalado.

    El escalado es imprescindible: MedInc se mueve en decenas de miles de
    dólares (0.5-15) y las coordenadas en decenas de grados. Sin escalar,
    la distancia euclídea quedaría dominada por latitud y longitud.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE),
            ),
        ]
    )


def build_classifier_candidates() -> dict[str, Pipeline]:
    """Candidatos supervisados. Todos incluyen su propio escalador."""
    return {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "KNeighbors": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=15)),
            ]
        ),
    }


def compare_classifiers(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[str, dict[str, dict[str, float]]]:
    """Compara candidatos con validación cruzada estratificada de 5 folds."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores: dict[str, dict[str, float]] = {}

    for name, pipeline in build_classifier_candidates().items():
        fold_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
        )
        scores[name] = {
            "accuracy_mean": float(fold_scores.mean()),
            "accuracy_std": float(fold_scores.std()),
        }
        print(f"  {name:<20} accuracy CV = {fold_scores.mean():.4f} ± {fold_scores.std():.4f}")

    best_name = max(scores, key=lambda name: scores[name]["accuracy_mean"])
    return best_name, scores


def cluster_profiles(
    df: pd.DataFrame, labels: np.ndarray, centroids_latlon: np.ndarray
) -> list[dict]:
    """Resume cada cluster: tamaño, renta y centro geográfico."""
    labeled = df.assign(cluster=labels)
    grouped = labeled.groupby("cluster", observed=True)
    profiles = []

    for cluster_id, group in grouped:
        profiles.append(
            {
                "cluster": int(cluster_id),
                "n": int(len(group)),
                "share_pct": round(100 * len(group) / len(labeled), 2),
                "medinc_mean": round(float(group["MedInc"].mean()), 3),
                "medinc_median": round(float(group["MedInc"].median()), 3),
                "latitude_mean": round(float(group["Latitude"].mean()), 3),
                "longitude_mean": round(float(group["Longitude"].mean()), 3),
                "centroid_latitude": round(float(centroids_latlon[cluster_id, 0]), 3),
                "centroid_longitude": round(float(centroids_latlon[cluster_id, 1]), 3),
            }
        )

    return profiles


def evaluate_classifier(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calcula las métricas principales sobre el conjunto de test."""
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )

    return {
        "y_pred": y_pred,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def save_artifacts(
    kmeans_pipeline: Pipeline,
    classifier: Pipeline,
    df: pd.DataFrame,
    labels: np.ndarray,
    test_labels: np.ndarray,
    train_index: pd.Index,
    test_index: pd.Index,
    best_name: str,
    cv_scores: dict,
    metrics: dict,
    profiles: list[dict],
) -> None:
    """Guarda modelos, métricas y el dataset etiquetado."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Los dos artefactos son autocontenidos: cada uno lleva su StandardScaler.
    joblib.dump(kmeans_pipeline, KMEANS_PATH)
    joblib.dump(classifier, CLASSIFIER_PATH)

    # Dataset etiquetado: posición original, split y cluster.
    labeled = df.copy()
    labeled["split"] = "train"
    labeled.loc[test_index, "split"] = "test"
    labeled["cluster"] = pd.NA
    labeled.loc[train_index, "cluster"] = labels
    labeled.loc[test_index, "cluster"] = test_labels
    labeled["cluster"] = labeled["cluster"].astype(int).astype("category")
    labeled.to_csv(PROCESSED_PATH, index=True, index_label="row_id")

    metadata = {
        "project": "California Housing K-Means",
        "features": FEATURES,
        "n_clusters": N_CLUSTERS,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train": int(len(train_index)),
        "n_test": int(len(test_index)),
        "kmeans": {
            "params": {
                "n_clusters": N_CLUSTERS,
                "n_init": 10,
                "random_state": RANDOM_STATE,
            },
            "inertia_train": float(kmeans_pipeline.named_steps["kmeans"].inertia_),
            "silhouette_test": float(
                silhouette_score(
                    kmeans_pipeline.named_steps["scaler"].transform(
                        df.loc[test_index]
                    ),
                    test_labels,
                )
            ),
        },
        "classifier": {
            "selected": best_name,
            "cv_accuracy": cv_scores,
            "test_accuracy": metrics["accuracy"],
            "test_balanced_accuracy": metrics["balanced_accuracy"],
            "test_f1_macro": metrics["f1_macro"],
            "confusion_matrix": metrics["confusion_matrix"],
        },
        "cluster_profiles": profiles,
        "versions": {
            "python": ".".join(map(str, __import__("sys").version_info[:3])),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    print("1. Cargando datos...")
    df = load_data()
    print(f"   {len(df):,} filas | features: {', '.join(FEATURES)}")

    print("2. Dividiendo en train/test...")
    train_index, test_index = split_data(df)
    X_train = df.loc[train_index]
    X_test = df.loc[test_index]
    print(f"   train={len(X_train):,} | test={len(X_test):,}")

    print("3. Entrenando K-Means (k=6)...")
    kmeans_pipeline = build_kmeans()
    kmeans_pipeline.fit(X_train)
    labels = kmeans_pipeline.predict(X_train)
    test_labels = kmeans_pipeline.predict(X_test)

    X_train_scaled = kmeans_pipeline.named_steps["scaler"].transform(X_train)
    X_test_scaled = kmeans_pipeline.named_steps["scaler"].transform(X_test)
    print(f"   inertia train = {kmeans_pipeline.named_steps['kmeans'].inertia_:,.1f}")
    print(
        "   silhouette test = "
        f"{silhouette_score(X_test_scaled, test_labels):.4f}"
    )

    # Centroides en unidades originales para poder graficarlos.
    centroids_scaled = kmeans_pipeline.named_steps["kmeans"].cluster_centers_
    centroids_latlon = (
        kmeans_pipeline.named_steps["scaler"].inverse_transform(centroids_scaled)
    )

    print("4. Comparando clasificadores supervisados...")
    best_name, cv_scores = compare_classifiers(X_train, pd.Series(labels))
    print(f"   Seleccionado: {best_name}")

    print("5. Evaluando el clasificador sobre test...")
    classifier = build_classifier_candidates()[best_name]
    classifier.fit(X_train, labels)
    metrics = evaluate_classifier(classifier, X_test, pd.Series(test_labels))
    print(f"   accuracy={metrics['accuracy']:.4f} | "
          f"balanced={metrics['balanced_accuracy']:.4f} | "
          f"f1_macro={metrics['f1_macro']:.4f}")

    print("6. Guardando artefactos...")
    profiles = cluster_profiles(X_train, labels, centroids_latlon)
    save_artifacts(
        kmeans_pipeline=kmeans_pipeline,
        classifier=classifier,
        df=df,
        labels=labels,
        test_labels=test_labels,
        train_index=train_index,
        test_index=test_index,
        best_name=best_name,
        cv_scores=cv_scores,
        metrics=metrics,
        profiles=profiles,
    )
    print(f"   {KMEANS_PATH}")
    print(f"   {CLASSIFIER_PATH}")
    print(f"   {PROCESSED_PATH}")
    print(f"   {METADATA_PATH}")


if __name__ == "__main__":
    main()
