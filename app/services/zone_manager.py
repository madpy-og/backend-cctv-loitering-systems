"""
app.services.zone_manager — CRUD untuk zona bahaya (in-memory + persistensi SQLite).

Zona disimpan in-memory agar pipeline bisa akses tanpa I/O DB tiap frame.
Perubahan zona langsung diterapkan ke pipeline tanpa restart (FR-4.3).
"""

import json
import logging
import threading
from typing import Any

from sqlmodel import Session, select
from app.core.db import engine
from app.models.zone import Zone

logger = logging.getLogger(__name__)

class ZoneManager:
    """Mengelola CRUD zona bahaya dengan persistensi database SQLite.

    Thread-safe: zona bisa dibaca dari pipeline thread dan ditulis dari API thread.
    Menggunakan in-memory cache agar pipeline tidak bottleneck saat query tiap frame.
    """

    def __init__(self):
        # Cache in-memory: key = zone_id (int), value = Zone model
        self._zones: dict[int, Zone] = {}
        self._lock = threading.Lock()

    def load(self) -> None:
        """Muat semua zona dari database ke dalam cache in-memory."""
        with self._lock:
            try:
                with Session(engine) as session:
                    zones_db = session.exec(select(Zone)).all()
                    self._zones = {z.id: z for z in zones_db if z.id is not None}
                logger.info(f"Loaded {len(self._zones)} zones from database.")
            except Exception as e:
                logger.error(f"Failed to load zones from database: {e}")
                self._zones = {}

    def create_zone(self, name: str, points: list[dict[str, float]], is_active: bool = True) -> Zone:
        """Buat zona baru dan simpan ke database.

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

        coordinates_json = json.dumps(points)
        zone = Zone(name=name, coordinates=coordinates_json, is_active=is_active)

        with self._lock:
            with Session(engine) as session:
                session.add(zone)
                session.commit()
                session.refresh(zone)
                
            if zone.id is not None:
                self._zones[zone.id] = zone

        logger.info(f"Zone created: id={zone.id}, name={name}, points={len(points)}")
        return zone

    def get_zone(self, zone_id: int) -> Zone | None:
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
        zone_id: int,
        name: str | None = None,
        points: list[dict[str, float]] | None = None,
        is_active: bool | None = None,
    ) -> Zone | None:
        """Update zona yang sudah ada di database dan cache.

        Returns:
            Zona yang diupdate, atau None jika tidak ditemukan.
        """
        with self._lock:
            if zone_id not in self._zones:
                return None

            with Session(engine) as session:
                # Fetch fresh from DB
                zone = session.get(Zone, zone_id)
                if not zone:
                    return None

                if name is not None:
                    zone.name = name
                if points is not None:
                    if len(points) < 3:
                        raise ValueError("Zona memerlukan minimal 3 titik polygon.")
                    zone.coordinates = json.dumps(points)
                if is_active is not None:
                    zone.is_active = is_active

                session.add(zone)
                session.commit()
                session.refresh(zone)

                # Update cache
                self._zones[zone_id] = zone

        logger.info(f"Zone updated: id={zone_id}")
        return zone

    def delete_zone(self, zone_id: int) -> bool:
        """Hapus zona dari database dan cache.

        Returns:
            True jika zona berhasil dihapus, False jika tidak ditemukan.
        """
        with self._lock:
            if zone_id not in self._zones:
                return False

            with Session(engine) as session:
                zone = session.get(Zone, zone_id)
                if zone:
                    session.delete(zone)
                    session.commit()

            del self._zones[zone_id]

        logger.info(f"Zone deleted: id={zone_id}")
        return True
