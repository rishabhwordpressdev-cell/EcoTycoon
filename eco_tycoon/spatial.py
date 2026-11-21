from __future__ import annotations
import numpy as np
from scipy import ndimage
from typing import Dict


def fragmentation_metrics(grid: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate fragmentation and connectivity metrics from a habitat suitability grid.

    Parameters
    ----------
    grid : 2D numpy array
        Habitat suitability values (0-1)
    threshold : float
        Threshold to convert suitability to binary habitat/non-habitat

    Returns
    -------
    dict
        Dictionary containing fragmentation metrics
    """
    if grid.ndim != 2:
        raise ValueError("Grid must be 2-dimensional")

    # Binarize
    binary = (grid >= threshold).astype(int)

    # Label connected components (4-connectivity)
    labeled, num_patches = ndimage.label(binary, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    if num_patches == 0:
        return {
            "num_patches": 0,
            "total_habitat_cells": 0,
            "largest_patch_area": 0,
            "largest_patch_share": 0.0,
            "mean_patch_area": 0.0,
            "patch_density": 0.0,
            "edge_density": 0.0,
            "landscape_shape_index": 0.0,
            "connectivity_index": 0.0,
            "fragmentation_index": 1.0,
        }

    # Patch areas
    patch_areas = ndimage.sum(binary, labeled, range(1, num_patches + 1))
    total_habitat = int(patch_areas.sum())
    largest_patch = float(patch_areas.max())
    mean_patch = float(patch_areas.mean())

    # Largest patch share
    largest_patch_share = largest_patch / total_habitat if total_habitat > 0 else 0.0

    # Patch density (patches per 100 cells)
    total_cells = grid.size
    patch_density = (num_patches / total_cells) * 100

    # Edge density
    # Calculate edges using gradient
    edges_y = np.abs(np.diff(binary, axis=0))
    edges_x = np.abs(np.diff(binary, axis=1))
    total_edges = edges_y.sum() + edges_x.sum()
    edge_density = total_edges / total_cells if total_cells > 0 else 0.0

    # Landscape Shape Index (LSI)
    # LSI = (total edge) / (2 * sqrt(pi * total habitat area))
    if total_habitat > 0:
        lsi = total_edges / (2 * np.sqrt(np.pi * total_habitat))
    else:
        lsi = 0.0

    # Connectivity index (based on probability of two random habitat cells being connected)
    # Simplified: proportion of habitat in largest patch
    connectivity_index = largest_patch_share

    # Fragmentation index (inverse of connectivity, scaled)
    fragmentation_index = 1.0 - connectivity_index

    return {
        "num_patches": int(num_patches),
        "total_habitat_cells": total_habitat,
        "largest_patch_area": largest_patch,
        "largest_patch_share": round(largest_patch_share, 4),
        "mean_patch_area": round(mean_patch, 2),
        "patch_density": round(patch_density, 4),
        "edge_density": round(edge_density, 4),
        "landscape_shape_index": round(lsi, 4),
        "connectivity_index": round(connectivity_index, 4),
        "fragmentation_index": round(fragmentation_index, 4),
    }