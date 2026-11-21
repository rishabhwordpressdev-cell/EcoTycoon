from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple


def clean_occurrences(df: pd.DataFrame) -> pd.DataFrame:
    """Clean GBIF occurrence records."""
    if df.empty:
        return df

    # Required columns
    required = ["decimal_latitude", "decimal_longitude", "species"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = df.copy()

    # Remove records with missing coordinates
    cleaned = cleaned.dropna(subset=["decimal_latitude", "decimal_longitude"])

    # Remove invalid coordinates (zeros, out of bounds)
    cleaned = cleaned[
        (cleaned["decimal_latitude"].between(-90, 90))
        & (cleaned["decimal_longitude"].between(-180, 180))
        & ~((cleaned["decimal_latitude"] == 0) & (cleaned["decimal_longitude"] == 0))
    ]

    # Remove duplicates based on coordinates and species
    cleaned = cleaned.drop_duplicates(subset=["decimal_latitude", "decimal_longitude", "species"])

    # Remove records with high coordinate uncertainty (> 10km)
    if "coordinate_uncertainty" in cleaned.columns:
        cleaned = cleaned[
            (cleaned["coordinate_uncertainty"].isna())
            | (cleaned["coordinate_uncertainty"] <= 10000)
        ]

    return cleaned.reset_index(drop=True)


def spatial_thin_occurrences(
    df: pd.DataFrame, min_distance_km: float = 1.0
) -> pd.DataFrame:
    """Spatially thin occurrences to reduce sampling bias."""
    if df.empty or len(df) <= 1:
        return df

    # Simple grid-based thinning
    # Round coordinates to grid cells of approximately min_distance_km
    # 1 degree ~= 111 km at equator
    grid_size = min_distance_km / 111.0

    df = df.copy()
    df["grid_lat"] = (df["decimal_latitude"] / grid_size).astype(int)
    df["grid_lon"] = (df["decimal_longitude"] / grid_size).astype(int)

    # Keep one record per grid cell (first occurrence)
    thinned = df.drop_duplicates(subset=["grid_lat", "grid_lon"])

    return thinned.drop(columns=["grid_lat", "grid_lon"]).reset_index(drop=True)