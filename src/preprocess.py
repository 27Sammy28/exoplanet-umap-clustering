"""Preprocessing utilities for the PAGER exoplanet workflow.

The functions here are dependency-light and safe to run in Google Colab. They load
PHL/NASA catalogue tables, map NASA PSCompPars columns onto the common PAGER
feature schema, impute missing values, and apply the log transforms used by the
ranking scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

PHL_FEATURES = [
    "P_MASS",
    "P_RADIUS",
    "P_PERIOD",
    "P_SEMI_MAJOR_AXIS",
    "P_ECCENTRICITY",
    "S_TEMPERATURE",
    "S_MASS",
    "S_RADIUS",
    "S_LOG_LUM",
    "P_FLUX",
    "P_TEMP_EQUIL",
]

EXTENDED_PHL_FEATURES = PHL_FEATURES + ["P_GRAVITY", "P_DENSITY", "P_ESI"]

NASA_TO_COMMON = {
    "pl_bmasse": "P_MASS",
    "pl_rade": "P_RADIUS",
    "pl_orbper": "P_PERIOD",
    "pl_orbsmax": "P_SEMI_MAJOR_AXIS",
    "pl_orbeccen": "P_ECCENTRICITY",
    "st_teff": "S_TEMPERATURE",
    "st_mass": "S_MASS",
    "st_rad": "S_RADIUS",
    "pl_insol": "P_FLUX",
    "pl_eqt": "P_TEMP_EQUIL",
}

POSITIVE_LOG_COLUMNS = {
    "P_MASS",
    "P_RADIUS",
    "P_PERIOD",
    "P_SEMI_MAJOR_AXIS",
    "S_TEMPERATURE",
    "S_MASS",
    "S_RADIUS",
    "P_FLUX",
    "P_TEMP_EQUIL",
    "P_GRAVITY",
    "P_DENSITY",
}


@dataclass
class PreparedData:
    """Container returned by :func:`prepare_features`."""

    names: pd.Series
    raw_features: pd.DataFrame
    transformed: np.ndarray
    completeness: np.ndarray
    medians: pd.Series


def normalise_name(value: object) -> str:
    """Return a compact lower-case planet name for cross-catalogue matching."""

    text = str(value).lower().strip()
    return re.sub(r"[^a-z0-9]+", "", text)


def read_nasa_csv(path: str | Path) -> pd.DataFrame:
    """Read a NASA Exoplanet Archive CSV, skipping leading comment metadata."""

    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        skiprows = 0
        for line in handle:
            if line.startswith("#") or not line.strip():
                skiprows += 1
                continue
            break
    return pd.read_csv(path, skiprows=skiprows)


def phl_labels(df: pd.DataFrame, label_column: str = "P_HABITABLE") -> np.ndarray:
    """Convert PHL habitability labels into binary PHL-prioritised labels."""

    return (pd.to_numeric(df[label_column], errors="coerce").fillna(0) > 0).astype(int).to_numpy()


def map_nasa_to_common(nasa_raw: pd.DataFrame, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    """Map NASA PSCompPars columns into the common PAGER feature names."""

    mapping = NASA_TO_COMMON if mapping is None else mapping
    mapped = pd.DataFrame(index=nasa_raw.index)
    for source, target in mapping.items():
        mapped[target] = pd.to_numeric(nasa_raw[source], errors="coerce") if source in nasa_raw else np.nan
    mapped["S_LOG_LUM"] = np.nan
    return mapped


def prepare_features(
    df: pd.DataFrame,
    features: list[str] | None = None,
    names: pd.Series | None = None,
    medians: pd.Series | None = None,
) -> PreparedData:
    """Select, impute, log-transform, and return model-ready feature arrays."""

    features = PHL_FEATURES if features is None else features
    names = df.index.to_series().astype(str) if names is None else names.reset_index(drop=True)
    feature_df = df.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    completeness = feature_df.notna().mean(axis=1).to_numpy(float)
    if medians is None:
        medians = feature_df.median(numeric_only=True)
    filled = feature_df.fillna(medians).fillna(0.0)
    transformed = filled.to_numpy(float)

    for index, column in enumerate(features):
        if column not in POSITIVE_LOG_COLUMNS:
            continue
        values = transformed[:, index]
        positive = values[np.isfinite(values) & (values > 0)]
        min_positive = positive.min() if positive.size else 1e-9
        values = np.where(values > 0, values, min_positive)
        transformed[:, index] = np.log10(values)

    return PreparedData(
        names=names,
        raw_features=feature_df,
        transformed=transformed,
        completeness=completeness,
        medians=medians,
    )


def candidate_mask(labels: np.ndarray, raw_features: pd.DataFrame, min_features: int = 6) -> np.ndarray:
    """Return the PHL-positive candidates with enough observed features."""

    mask = labels == 1
    mask &= raw_features.notna().sum(axis=1).to_numpy() >= min_features
    return mask





def load_data():
    """
    Wrapper for ML pipeline compatibility.
    Uses real PAGER preprocessing pipeline.
    """

    # If you have a real dataset loader elsewhere, plug it here
    # Example assumption: you load a CSV manually

    import pandas as pd

    # CHANGE THIS PATH to your actual dataset file
    path = "data/raw/hwc.csv"

    df = pd.read_csv(path)

    prepared = prepare_features(df)

    return prepared.transformed
