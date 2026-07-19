"""
app.services.alert_logger — SQLite logging + snapshot saving untuk alert loitering.

Catat alert ke tabel `Alert` dalam database SQLite.
"""

import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from sqlmodel import Session, select, col

from app.core.db import engine
from app.models.alert import Alert
from app.services.behavior import AlertEvent

logger = logging.getLogger(__name__)

# JPEG quality sweet spot: quality 85
JPEG_QUALITY = 85

class AlertLogger:
    """Mengelola logging alert ke SQLite dan penyimpanan snapshot evidence.

    Thread-safe: alert bisa ditulis dari pipeline thread dan dibaca dari API thread.
    """

    def __init__(self, snapshots_dir: str = "data/snapshots"):
        self._snapshots_dir = Path(snapshots_dir)
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """Buat folder snapshot jika belum ada."""
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Alert logger initialized. Snapshots dir: {self._snapshots_dir}")

    def log_alert(
        self,
        event: AlertEvent,
        frame: np.ndarray | None = None,
    ) -> Alert:
        """Catat alert ke SQLite dan simpan snapshot.

        Args:
            event: AlertEvent dari BehaviorAnalyzer.
            frame: Frame dengan overlay (opsional, untuk snapshot evidence).

        Returns:
            Objek Alert yang dicatat ke database.
        """
        now = datetime.now()

        # Simpan snapshot jika frame tersedia
        snapshot_filename = None
        if frame is not None:
            # We don't have an alert ID yet since it's auto-increment,
            # so we generate a random string for the filename or use timestamp.
            timestamp_str = now.strftime("%Y%m%d_%H%M%S")
            snapshot_filename = f"alert_track{event.track_id}_frame{event.frame_number}_{timestamp_str}.jpg"
            self._save_snapshot(
                frame=frame,
                filename=snapshot_filename,
            )

        alert_db = Alert(
            # Kita simpan waktu trigger
            timestamp=now,
            track_id=event.track_id,
            duration=event.dwell_time_seconds,
            snapshot_path=snapshot_filename,
            zone_id=event.zone_id,
        )

        with self._lock:
            with Session(engine) as session:
                session.add(alert_db)
                session.commit()
                session.refresh(alert_db)

        logger.info(f"Alert logged: id={alert_db.id}, track={event.track_id}, "
                    f"zone={event.zone_name}, dwell={event.dwell_time_seconds:.2f}s")

        return alert_db

    def _save_snapshot(self, frame: np.ndarray, filename: str) -> None:
        """Simpan frame sebagai bukti visual (JPEG, quality 85)."""
        filepath = self._snapshots_dir / filename
        cv2.imwrite(
            str(filepath),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        logger.debug(f"Snapshot saved: {filepath}")

    def get_alerts(self, limit: int = 50, offset: int = 0) -> list[Alert]:
        """Ambil daftar alert terbaru (terbaru di depan)."""
        with self._lock:
            with Session(engine) as session:
                statement = select(Alert).order_by(col(Alert.timestamp).desc()).offset(offset).limit(limit)
                return list(session.exec(statement).all())

    def get_alert_by_id(self, alert_id: int) -> Alert | None:
        """Cari alert berdasarkan ID."""
        with self._lock:
            with Session(engine) as session:
                return session.get(Alert, alert_id)

    def get_snapshot_path(self, snapshot_filename: str) -> Path | None:
        """Dapatkan full path ke file snapshot."""
        if not snapshot_filename:
            return None
        filepath = self._snapshots_dir / snapshot_filename
        if filepath.exists():
            return filepath
        return None

    def get_total_alerts(self) -> int:
        """Total jumlah alert yang tercatat."""
        with self._lock:
            with Session(engine) as session:
                from sqlalchemy import func
                statement = select(func.count()).select_from(Alert)
                return session.exec(statement).one()
