"""
app.services.frame_streamer — Thread-safe frame buffer untuk MJPEG streaming.

Refactored from: loitering_system.py → draw_overlays() (lines 114-147).

Mengelola:
1. Overlay rendering (bounding box, track ID, timer, zona polygon) pada frame.
2. Buffer frame terbaru untuk dikonsumsi oleh MJPEG endpoint.
"""

import threading
import logging

import cv2
import numpy as np

from app.services.tracker import Track
from app.services.behavior import BehaviorAnalyzer
from app.services.zone_manager import ZoneManager

logger = logging.getLogger(__name__)

# Visual config (dari loitering_system.py lines 80-83)
BBOX_COLOR_NORMAL = (0, 255, 0)       # Hijau (BGR)
BBOX_COLOR_LOITERING = (0, 0, 255)    # Merah (BGR)
ZONE_COLOR = (255, 0, 0)              # Biru (BGR)
ZONE_ALPHA = 0.3


class FrameStreamer:
    """Buffer frame terbaru (dengan overlay) untuk MJPEG streaming.

    Producer: pipeline thread (update setiap frame).
    Consumer: HTTP client via /video_feed endpoint.
    """

    def __init__(self):
        self._frame: np.ndarray | None = None
        self._jpeg_bytes: bytes | None = None
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def update_frame(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        behavior_analyzer: BehaviorAnalyzer,
        zone_manager: ZoneManager,
        current_frame: int,
        fps: float,
    ) -> np.ndarray:
        """Render overlay pada frame lalu simpan ke buffer.

        Args:
            frame: Frame raw dari camera.
            tracks: List confirmed tracks.
            behavior_analyzer: Untuk mendapatkan state loitering per track.
            zone_manager: Untuk mendapatkan zona aktif.
            current_frame: Nomor frame saat ini.
            fps: FPS video.

        Returns:
            Frame yang sudah di-overlay (untuk snapshot evidence).
        """
        output_frame = self._draw_overlays(
            frame=frame.copy(),
            tracks=tracks,
            behavior_analyzer=behavior_analyzer,
            zone_manager=zone_manager,
            current_frame=current_frame,
            fps=fps,
        )

        # Encode ke JPEG untuk MJPEG streaming
        _, jpeg = cv2.imencode(".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        with self._condition:
            self._frame = output_frame
            self._jpeg_bytes = jpeg.tobytes()
            self._condition.notify_all()

        return output_frame

    def get_jpeg_bytes(self) -> bytes | None:
        """Ambil frame terbaru sebagai JPEG bytes (non-blocking)."""
        with self._lock:
            return self._jpeg_bytes

    def get_frame(self) -> np.ndarray | None:
        """Ambil frame terbaru sebagai numpy array (non-blocking)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def wait_for_frame(self, timeout: float = 1.0) -> bytes | None:
        """Tunggu sampai frame baru tersedia (blocking dengan timeout).

        Dipakai oleh MJPEG generator agar tidak busy-loop.
        """
        with self._condition:
            self._condition.wait(timeout=timeout)
            return self._jpeg_bytes

    def generate_mjpeg(self):
        """Generator untuk MJPEG streaming (dipakai oleh StreamingResponse).

        Yields:
            Bytes MJPEG frame dengan boundary header.
        """
        while True:
            jpeg_bytes = self.wait_for_frame(timeout=2.0)
            if jpeg_bytes is None:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg_bytes
                + b"\r\n"
            )

    def _draw_overlays(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        behavior_analyzer: BehaviorAnalyzer,
        zone_manager: ZoneManager,
        current_frame: int,
        fps: float,
    ) -> np.ndarray:
        """Gambar overlay pada frame: zona bahaya, bounding box, track ID, timer.

        Refactored dari loitering_system.py draw_overlays() (lines 114-147).
        """
        frame_h, frame_w = frame.shape[:2]

        # --- Gambar zona bahaya ---
        active_zones = zone_manager.get_active_zones()
        for zone in active_zones:
            polygon = zone.to_polygon_array(frame_w, frame_h)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon.reshape(-1, 2)], ZONE_COLOR)
            frame = cv2.addWeighted(overlay, ZONE_ALPHA, frame, 1 - ZONE_ALPHA, 0)
            cv2.polylines(frame, [polygon.reshape(-1, 2)], True, ZONE_COLOR, 2)

        # --- Gambar bounding box, track ID, timer ---
        states = behavior_analyzer.get_all_states()

        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            current_color = BBOX_COLOR_NORMAL

            # Cek apakah track ini sedang loitering di salah satu zona
            is_track_loitering = False
            track_elapsed_sec = 0.0
            track_in_zone = False

            for zone in active_zones:
                state = states.get((track.track_id, zone.id))
                if state and state.is_loitering:
                    is_track_loitering = True
                if state and state.in_zone and state.start_frame != -1:
                    track_in_zone = True
                    elapsed = current_frame - state.start_frame
                    elapsed_sec = elapsed / fps if fps > 0 else 0
                    track_elapsed_sec = max(track_elapsed_sec, elapsed_sec)

            if is_track_loitering:
                current_color = BBOX_COLOR_LOITERING

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), current_color, 2)

            # Track ID
            cv2.putText(
                frame, str(track.track_id),
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2,
            )

            # Timer durasi (jika sedang di dalam zona)
            if track_in_zone:
                timer_text = f"{track_elapsed_sec:.1f}s"
                cv2.putText(
                    frame, timer_text,
                    (x1, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, current_color, 2,
                )

        return frame
