"""
app.services.zone_manager — CRUD untuk zona bahaya (in-memory + persistensi JSON).

Zona disimpan in-memory agar pipeline bisa akses tanpa I/O tiap frame.
Perubahan zona langsung diterapkan ke pipeline tanpa restart (FR-4.3).
"""

import json
import uuid
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Zone:
    """Representasi satu zona bahaya."""

    def __init__(
        self,
        zone_id: str,
        name: str,
        points: list[dict[str, float]],
        is_active: bool = True,
        created_at: str | None = None,
    ):
        self.id = zone_id
        self.name = name
        self.points = points  # List of {"x": float, "y": float}
        self.is_active = is_active
        self.created_at = created_at or datetime.now().isoformat()

    def to_polygon_array(self, frame_width: int, frame_height: int) -> np.ndarray:
        """Konversi points ke numpy polygon array untuk cv2.pointPolygonTest.

        Points disimpan sebagai koordinat pixel.
        Returns:
            np.ndarray shape (1, N, 2) cocok untuk cv2.fillPoly / cv2.pointPolygonTest.
        """
        pts = [(int(p["x"]), int(p["y"])) for p in self.points]
        return np.array([pts], np.int32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "points": self.points,
            "is_active": self.is_active,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Zone":
        return cls(
            zone_id=data["id"],
            name=data["name"],
            points=data["points"],
            is_active=data.get("is_active", True),
            created_at=data.get("created_at"),
        )


class ZoneManager:
    """Mengelola CRUD zona bahaya dengan persistensi file JSON.

    Thread-safe: zona bisa dibaca dari pipeline thread dan ditulis dari API thread.
    """

    def __init__(self, config_path: str = "data/zones.json"):
        self._config_path = Path(config_path)
        self._zones: dict[str, Zone] = {}
        self._lock = threading.Lock()

    def load(self) -> None:
        """Muat zona dari file JSON (jika ada)."""
        if self._config_path.exists():
            try:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                with self._lock:
                    self._zones = {
                        z["id"]: Zone.from_dict(z) for z in data.get("zones", [])
                    }
                logger.info(f"Loaded {len(self._zones)} zones from {self._config_path}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse zones config: {e}. Starting with empty zones.")
                self._zones = {}
        else:
            logger.info(f"No zones config found at {self._config_path}. Starting with empty zones.")

    def _save(self) -> None:
        """Simpan zona ke file JSON. Harus dipanggil dalam context self._lock."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"zones": [z.to_dict() for z in self._zones.values()]}
        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def create_zone(self, name: str, points: list[dict[str, float]], is_active: bool = True) -> Zone:
        """Buat zona baru.

        Args:
            name: Nama zona.
            points: List titik polygon [{"x": float, "y": float}, ...]. Minimal 3 titik.
            is_active: Apakah zona langsung aktif.

        Returns:
            Zona yang baru dibuat.

        Raises:
            ValueError: Jika points kurang dari 3.
        """
        if len(points) < 3:
            raise ValueError("Zona memerlukan minimal 3 titik polygon.")

        zone_id = str(uuid.uuid4())
        zone = Zone(zone_id=zone_id, name=name, points=points, is_active=is_active)

        with self._lock:
            self._zones[zone_id] = zone
            self._save()

        logger.info(f"Zone created: id={zone_id}, name={name}, points={len(points)}")
        return zone

    def get_zone(self, zone_id: str) -> Zone | None:
        """Ambil zona berdasarkan ID."""
        with self._lock:
            return self._zones.get(zone_id)

    def list_zones(self) -> list[Zone]:
        """List semua zona."""
        with self._lock:
            return list(self._zones.values())

    def get_active_zones(self) -> list[Zone]:
        """List zona yang aktif saja (dipakai oleh pipeline setiap frame)."""
        with self._lock:
            return [z for z in self._zones.values() if z.is_active]

    def update_zone(
        self,
        zone_id: str,
        name: str | None = None,
        points: list[dict[str, float]] | None = None,
        is_active: bool | None = None,
    ) -> Zone | None:
        """Update zona yang sudah ada.

        Returns:
            Zona yang diupdate, atau None jika tidak ditemukan.
        """
        with self._lock:
            zone = self._zones.get(zone_id)
            if zone is None:
                return None

            if name is not None:
                zone.name = name
            if points is not None:
                if len(points) < 3:
                    raise ValueError("Zona memerlukan minimal 3 titik polygon.")
                zone.points = points
            if is_active is not None:
                zone.is_active = is_active

            self._save()

        logger.info(f"Zone updated: id={zone_id}")
        return zone

    def delete_zone(self, zone_id: str) -> bool:
        """Hapus zona.

        Returns:
            True jika zona berhasil dihapus, False jika tidak ditemukan.
        """
        with self._lock:
            if zone_id not in self._zones:
                return False
            del self._zones[zone_id]
            self._save()

        logger.info(f"Zone deleted: id={zone_id}")
        return True
