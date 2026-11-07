"""Inferencia de producción para el sentimiento de las reseñas de Harbor House Café.

Carga el modelo ``nlptown`` fijado una sola vez, puntúa cada reseña de
``data/raw/reviews.csv`` y escribe el dataset enriquecido en
``data/processed/reviews_with_sentiment.csv``.

Los pesos se descargan de Hugging Face en runtime y se cachean fuera del
repositorio. Nunca se guardan en este proyecto.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from transformers.utils import logging as transformers_logging

# Anclamos las rutas a la raíz del repositorio, no al directorio desde el
# que se ejecute el script. Así `python src/app.py` funciona desde cualquier CWD.
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "reviews.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "reviews_with_sentiment.csv"

# Modelo y revisión inmutable: el resultado no cambia si actualizan el repo.
MODEL_ID = "nlptown/bert-base-multilingual-uncased-sentiment"
# Commit inmutable de Hugging Face: no usar "main" ni "latest".
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"

# Parámetros de inferencia. 500 reseñas cortas se procesan en CPU sin problema.
EXPECTED_ROWS = 500
BATCH_SIZE = 32
MAX_LENGTH = 512

# Traducimos la salida ordinal del modelo (1-5 estrellas) a tres bandas.
STAR_TO_BAND = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}
# Orden estable para los resúmenes y las tablas.
SENTIMENT_ORDER = ["positive", "neutral", "negative"]
# Columnas exactas que se escriben en el CSV de producción.
OUTPUT_COLUMNS = [
    "review_id",
    "rating",
    "review_text",
    "predicted_stars",
    "predicted_label",
    "predicted_sentiment",
    "confidence",
    "expected_stars",
    "prob_1_star",
    "prob_2_star",
    "prob_3_star",
    "prob_4_star",
    "prob_5_star",
    "star_implied_sentiment",
    "agrees_with_rating",
    "is_false_negative",
    "model_id",
    "model_revision",
]


def load_reviews(path: Path = RAW_PATH) -> pd.DataFrame:
    """Carga y limpia mínimamente las reseñas de origen."""
    if not path.exists():
        raise FileNotFoundError(f"Raw reviews not found: {path}")

    # Validamos el contrato de columnas antes de transformar nada.
    df = pd.read_csv(path)
    expected_columns = {"review_id", "rating", "review_text"}
    missing_columns = expected_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {sorted(missing_columns)}")

    # Normalizamos tipos y espacios; no cambiamos el contenido semántico.
    df = df.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="raise").astype(int)
    df["review_text"] = (
        df["review_text"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    )

    # Controles de calidad: tamaño esperado, IDs únicos, ratings válidos y texto.
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} reviews, found {len(df)}")
    if df["review_id"].duplicated().any():
        raise ValueError("Duplicated review_id values found")
    if not df["rating"].between(1, 5).all():
        raise ValueError("Ratings must be between 1 and 5")
    if df["review_text"].isna().any() or (df["review_text"] == "").any():
        raise ValueError("Empty review_text values found")

    return df


def load_sentiment_pipeline():
    """Carga el modelo fijado una vez y devuelve un pipeline reutilizable."""
    transformers_logging.set_verbosity_error()

    # Descarga (o reutiliza la caché) del tokenizer y del modelo en la
    # revisión fijada. Los pesos quedan en ~/.cache/huggingface, no en el repo.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )

    # El pipeline se crea una sola vez y se reutiliza para todas las reseñas.
    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # CPU: 500 short reviews run in well under a minute.
        truncation=True,
        max_length=MAX_LENGTH,
    )


def _parse_prediction(predictions: list[dict]) -> dict:
    """Convierte la salida del pipeline en estrellas, banda, confianza y probabilidades."""
    # top_k=5 devuelve las cinco etiquetas: construimos estrella -> probabilidad.
    probabilities = {
        int(item["label"].split()[0]): float(item["score"]) for item in predictions
    }
    # La etiqueta con mayor score es la estrella predicha.
    best = max(predictions, key=lambda item: item["score"])
    predicted_stars = int(best["label"].split()[0])

    # expected_stars es una media ponderada: útil para ver la incertidumbre
    # sin colapsar todo a la clase ganadora.
    parsed = {
        "predicted_stars": predicted_stars,
        "predicted_label": best["label"],
        "confidence": float(best["score"]),
        "expected_stars": sum(stars * probabilities[stars] for stars in range(1, 6)),
    }
    for stars in range(1, 6):
        parsed[f"prob_{stars}_star"] = probabilities[stars]
    return parsed


def score_reviews(df: pd.DataFrame, sentiment_pipeline) -> pd.DataFrame:
    """Puntúa todas las reseñas en una sola llamada batch y enriquece el dataframe."""
    # Una única llamada en batch: el modelo no se recarga por reseña.
    predictions = sentiment_pipeline(
        df["review_text"].tolist(),
        batch_size=BATCH_SIZE,
        truncation=True,
        max_length=MAX_LENGTH,
        top_k=5,
    )

    # Parseamos cada salida y la unimos al dataframe original por posición.
    parsed = pd.DataFrame([_parse_prediction(row) for row in predictions])
    scored = pd.concat([df.reset_index(drop=True), parsed], axis=1)

    # Derivamos la banda del modelo, la banda implícita por estrellas y los
    # desacuerdos. Un falso negativo estricto es 4-5★ humano -> 1-2★ modelo.
    scored["predicted_sentiment"] = scored["predicted_stars"].map(STAR_TO_BAND)
    scored["star_implied_sentiment"] = scored["rating"].map(STAR_TO_BAND)
    scored["agrees_with_rating"] = (
        scored["predicted_sentiment"] == scored["star_implied_sentiment"]
    )
    scored["is_false_negative"] = (scored["rating"] >= 4) & (
        scored["predicted_stars"] <= 2
    )
    scored["model_id"] = MODEL_ID
    scored["model_revision"] = MODEL_REVISION

    # Validaciones antes de devolver el resultado.
    if len(scored) != len(df):
        raise ValueError("Prediction count does not match review count")
    if scored["predicted_stars"].isna().any():
        raise ValueError("Some reviews did not receive a prediction")
    if not scored["predicted_stars"].between(1, 5).all():
        raise ValueError("Predicted stars must be between 1 and 5")

    return scored[OUTPUT_COLUMNS]


def write_outputs(scored: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    # Creamos data/processed si no existe y escribimos en UTF-8.
    path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    # Carga de datos y modelo: una sola instanciación por ejecución.
    df = load_reviews()
    print(f"Loaded {len(df)} reviews from {RAW_PATH}")

    sentiment_pipeline = load_sentiment_pipeline()
    print(f"Model: {MODEL_ID} @ {MODEL_REVISION}")

    # Inferencia y escritura del artefacto de producción.
    scored = score_reviews(df, sentiment_pipeline)
    write_outputs(scored)

    # Resumen para consola: distribución, calibración y falsos negativos.
    counts = (
        scored["predicted_sentiment"]
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
    )
    print(f"Predicted sentiment: {counts.to_dict()}")
    print(f"Mean human rating: {scored['rating'].mean():.3f}")
    print(f"Mean predicted stars: {scored['predicted_stars'].mean():.3f}")
    print(f"Mean expected stars: {scored['expected_stars'].mean():.3f}")
    print(f"Agreement with star-implied band: {scored['agrees_with_rating'].mean():.1%}")
    print(f"Strict false negatives: {int(scored['is_false_negative'].sum())}")
    print(f"Wrote {len(scored)} rows to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
