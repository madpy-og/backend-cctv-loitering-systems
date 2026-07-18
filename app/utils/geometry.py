"""
app.utils.geometry — Geometric helper functions.

Refactored from: loitering_system.py → point_in_polygon() and get_centroid().
"""

import cv2
import numpy as np


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    """Mengecek apakah suatu titik (centroid) berada di dalam poligon.

    Args:
        point: Tuple (x, y) koordinat titik yang akan dicek.
        polygon: Numpy array polygon dengan shape (N, 2) atau (1, N, 2).

    Returns:
        True jika titik berada di dalam atau di tepi polygon.
    """
    return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0


def get_centroid(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Menghitung titik tengah (centroid) dari bounding box.

    Args:
        bbox: Tuple (x1, y1, x2, y2) bounding box.

    Returns:
        Tuple (cx, cy) koordinat centroid.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)
