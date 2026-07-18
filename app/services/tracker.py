"""
app.services.tracker — DeepSORT tracking wrapper.

Refactored from: loitering_system.py → DeepSort init (line 171) dan
tracker.update_tracks (line 222).
"""

import logging
from typing import Any

import numpy as np

from app.services.detector import Detection

logger = logging.getLogger(__name__)


class Track:
    """Representasi satu track dari DeepSORT, dinormalisasi untuk konsumsi internal."""

    __slots__ = ("track_id", "bbox", "is_confirmed_flag", "_raw_track")

    def __init__(self, track_id: int, bbox: tuple[int, int, int, int],
                 is_confirmed_flag: bool, raw_track: Any = None):
        """
        Args:
            track_id: ID unik track (persisten antar-frame).
            bbox: (x1, y1, x2, y2) bounding box dalam koordinat pixel.
            is_confirmed_flag: Apakah track sudah dikonfirmasi (min_hits terpenuhi).
            raw_track: Referensi ke objek track asli dari DeepSORT (opsional).
        """
        self.track_id = track_id
        self.bbox = bbox
        self.is_confirmed_flag = is_confirmed_flag
        self._raw_track = raw_track

    def is_confirmed(self) -> bool:
        """Apakah track ini sudah confirmed (bukan tentative)."""
        return self.is_confirmed_flag


class TrackerService:
    """Wrapper untuk DeepSORT multi-object tracker."""

    def __init__(self, max_age: int = 30, n_init: int = 3):
        """
        Args:
            max_age: Berapa frame objek bisa hilang sebelum ID-nya dihapus.
            n_init: Berapa deteksi berturut-turut untuk membuat track baru.
        """
        self._max_age = max_age
        self._n_init = n_init
        self._tracker: Any = None

    def initialize(self) -> None:
        """Inisialisasi DeepSORT tracker.

        Raises:
            ImportError: Jika deep_sort_realtime belum terinstall.
        """
        from deep_sort_realtime.deepsort_tracker import DeepSort

        logger.info(f"Initializing DeepSORT tracker: max_age={self._max_age}, n_init={self._n_init}")
        self._tracker = DeepSort(max_age=self._max_age, n_init=self._n_init)
        logger.info("DeepSORT tracker initialized.")

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        """Update tracker dengan deteksi baru dan return list tracks terkini.

        Args:
            detections: List hasil deteksi dari DetectorService.
            frame: Frame gambar saat ini (dipakai DeepSORT untuk feature extraction).

        Returns:
            List Track yang aktif saat ini.
        """
        if self._tracker is None:
            raise RuntimeError("Tracker belum diinisialisasi. Panggil initialize() terlebih dahulu.")

        # Konversi deteksi ke format DeepSORT: list of ([x,y,w,h], conf, class_name)
        deepsort_detections = [d.to_deepsort_format() for d in detections]

        # Update tracker
        raw_tracks = self._tracker.update_tracks(deepsort_detections, frame=frame)

        # Konversi ke internal Track objects
        tracks: list[Track] = []
        for raw_track in raw_tracks:
            if not raw_track.is_confirmed():
                continue

            ltrb = raw_track.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

            tracks.append(Track(
                track_id=raw_track.track_id,
                bbox=(x1, y1, x2, y2),
                is_confirmed_flag=True,
                raw_track=raw_track,
            ))

        return tracks

    def reset(self) -> None:
        """Reset tracker state (misal saat ganti video source)."""
        if self._tracker is not None:
            self.initialize()
