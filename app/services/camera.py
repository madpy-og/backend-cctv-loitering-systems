"""
app.services.camera — Camera/video source manager.

Mendukung: file video path, webcam index (0), atau RTSP URL.
Refactored from: loitering_system.py → get_video_properties() dan cv2.VideoCapture logic.
"""

import cv2
import threading
import logging

logger = logging.getLogger(__name__)


class CameraService:
    """Mengelola koneksi ke sumber video (file, webcam, atau RTSP).

    Thread-safe: bisa di-read dari pipeline thread dan di-release dari main thread.
    """

    def __init__(self, source: str | int = 0):
        """
        Args:
            source: Path ke file video, index webcam (0, 1, ...),
                    atau URL RTSP. String numerik ("0") akan dikonversi ke int.
        """
        # Konversi string numerik ke int (misal "0" -> 0 untuk webcam)
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self._source = source
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._is_opened = False

    def open(self) -> bool:
        """Membuka koneksi ke sumber video.

        Returns:
            True jika berhasil membuka.
        """
        with self._lock:
            self._cap = cv2.VideoCapture(self._source)
            self._is_opened = self._cap.isOpened()
            if self._is_opened:
                logger.info(f"Camera opened: source={self._source}, "
                            f"resolution={self.width}x{self.height}, "
                            f"fps={self.fps:.1f}")
            else:
                logger.error(f"Failed to open camera: source={self._source}")
            return self._is_opened

    def read_frame(self):
        """Membaca satu frame dari sumber video.

        Returns:
            numpy.ndarray frame jika berhasil, None jika gagal atau video habis.
        """
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return None
            ret, frame = self._cap.read()
            if not ret:
                return None
            return frame

    def release(self) -> None:
        """Melepas koneksi ke sumber video."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
                self._is_opened = False
                logger.info("Camera released.")

    @property
    def width(self) -> int:
        """Lebar frame video dalam pixel."""
        if self._cap:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return 0

    @property
    def height(self) -> int:
        """Tinggi frame video dalam pixel."""
        if self._cap:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return 0

    @property
    def fps(self) -> float:
        """FPS sumber video."""
        if self._cap:
            return self._cap.get(cv2.CAP_PROP_FPS)
        return 0.0

    @property
    def total_frames(self) -> int:
        """Total jumlah frame (hanya relevan untuk file video, 0 untuk live)."""
        if self._cap:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0

    @property
    def is_opened(self) -> bool:
        return self._is_opened

    @property
    def source(self) -> str | int:
        return self._source
