"""
app.services.alert_logger — CSV logging + snapshot saving untuk alert loitering.

Refactored from: loitering_system.py → save_snapshot() (lines 150-158) dan
CSV logging logic (lines 312-316).
"""

import csv
import os
import uuid
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from app.services.behavior import AlertEvent

logger = logging.getLogger(__name__)

# Kolom CSV log
CSV_COLUMNS = [
    "alert_id",
    "track_id",
    "zone_id",
    "zone_name",
    "timestamp_masuk_zona",
    "timestamp_alert_triggered",
    "durasi_detik",
    "koordinat_centroid",
    "snapshot_filename",
]

# JPEG quality sweet spot: quality 85, hampir tidak kelihatan bedanya
# secara visual tapi ukuran file lebih kecil dari default (95)
JPEG_QUALITY = 85


class AlertLogger:
    """Mengelola logging alert ke CSV dan penyimpanan snapshot evidence.

    Thread-safe: alert bisa ditulis dari pipeline thread dan dibaca dari API thread.
    """

    def __init__(
        self,
        log_path: str = "data/logs/alerts.csv",
        snapshots_dir: str = "data/snapshots",
    ):
        self._log_path = Path(log_path)
        self._snapshots_dir = Path(snapshots_dir)
        self._lock = threading.Lock()
        self._alerts_cache: list[dict] = []  # In-memory cache untuk query cepat

    def initialize(self) -> None:
        """Buat folder dan file CSV jika belum ada, muat existing alerts ke cache."""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Muat existing alerts dari CSV ke cache
        if self._log_path.exists():
            try:
                with open(self._log_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    self._alerts_cache = list(reader)
                logger.info(f"Loaded {len(self._alerts_cache)} existing alerts from {self._log_path}")
            except Exception as e:
                logger.warning(f"Failed to load alerts CSV: {e}")
                self._alerts_cache = []
        else:
            # Buat file CSV baru dengan header
            with open(self._log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
            logger.info(f"Created new alerts log: {self._log_path}")

    def log_alert(
        self,
        event: AlertEvent,
        frame: np.ndarray | None = None,
    ) -> dict:
        """Catat alert ke CSV dan simpan snapshot.

        Args:
            event: AlertEvent dari BehaviorAnalyzer.
            frame: Frame dengan overlay (opsional, untuk snapshot evidence).

        Returns:
            Dict representasi alert yang dicatat.
        """
        alert_id = str(uuid.uuid4())
        now = datetime.now()
        timestamp_alert = now.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_masuk = (now - timedelta(seconds=event.dwell_time_seconds)).strftime("%Y-%m-%d %H:%M:%S")

        # Simpan snapshot jika frame tersedia
        snapshot_filename = ""
        if frame is not None:
            snapshot_filename = self._save_snapshot(
                frame=frame,
                alert_id=alert_id,
                track_id=event.track_id,
                frame_number=event.frame_number,
            )

        alert_record = {
            "alert_id": alert_id,
            "track_id": str(event.track_id),
            "zone_id": event.zone_id,
            "zone_name": event.zone_name,
            "timestamp_masuk_zona": timestamp_masuk,
            "timestamp_alert_triggered": timestamp_alert,
            "durasi_detik": f"{event.dwell_time_seconds:.2f}",
            "koordinat_centroid": f"({event.centroid[0]:.1f}, {event.centroid[1]:.1f})",
            "snapshot_filename": snapshot_filename,
        }

        with self._lock:
            # Append ke CSV
            with open(self._log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(alert_record)

            # Append ke in-memory cache
            self._alerts_cache.append(alert_record)

        logger.info(f"Alert logged: id={alert_id}, track={event.track_id}, "
                     f"zone={event.zone_name}, dwell={event.dwell_time_seconds:.2f}s")

        return alert_record

    def _save_snapshot(
        self,
        frame: np.ndarray,
        alert_id: str,
        track_id: int,
        frame_number: int,
    ) -> str:
        """Simpan frame sebagai bukti visual (JPEG, quality 85).

        Returns:
            Nama file snapshot (relatif, tanpa folder path).
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alert_{alert_id[:8]}_track{track_id}_frame{frame_number}_{timestamp_str}.jpg"
        filepath = self._snapshots_dir / filename

        cv2.imwrite(
            str(filepath),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )

        logger.debug(f"Snapshot saved: {filepath}")
        return filename

    def get_alerts(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Ambil daftar alert terbaru (terbaru di depan).

        Args:
            limit: Jumlah maksimum alert.
            offset: Offset dari alert terbaru.

        Returns:
            List dict alert, urut dari terbaru.
        """
        with self._lock:
            # Reverse agar terbaru di depan
            reversed_cache = list(reversed(self._alerts_cache))
            return reversed_cache[offset:offset + limit]

    def get_alert_by_id(self, alert_id: str) -> dict | None:
        """Cari alert berdasarkan ID."""
        with self._lock:
            for alert in self._alerts_cache:
                if alert.get("alert_id") == alert_id:
                    return alert
        return None

    def get_snapshot_path(self, snapshot_filename: str) -> Path | None:
        """Dapatkan full path ke file snapshot.

        Returns:
            Path ke file snapshot, atau None jika tidak ditemukan.
        """
        if not snapshot_filename:
            return None
        filepath = self._snapshots_dir / snapshot_filename
        if filepath.exists():
            return filepath
        return None

    def get_total_alerts(self) -> int:
        """Total jumlah alert yang tercatat."""
        with self._lock:
            return len(self._alerts_cache)
