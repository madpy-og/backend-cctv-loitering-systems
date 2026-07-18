"""
app.services.detector — YOLOv8 NCNN inference wrapper.

Refactored from: loitering_system.py → YOLO model loading (line 170) dan
inference loop (lines 212-219).
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Detection:
    """Representasi satu deteksi objek."""

    __slots__ = ("bbox", "confidence", "class_id")

    def __init__(self, bbox: tuple[int, int, int, int], confidence: float, class_id: int = 0):
        """
        Args:
            bbox: (x1, y1, x2, y2) bounding box dalam koordinat pixel.
            confidence: Skor kepercayaan (0.0 - 1.0).
            class_id: ID kelas objek (0 = person).
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id

    def to_deepsort_format(self) -> tuple[list[int], float, str]:
        """Konversi ke format yang diharapkan DeepSORT: ([x, y, w, h], conf, class_name)."""
        x1, y1, x2, y2 = self.bbox
        return ([x1, y1, x2 - x1, y2 - y1], self.confidence, "person")


class DetectorService:
    """Wrapper untuk YOLOv8 inference.

    Mendukung model NCNN (format yang dioptimalkan untuk edge/Raspberry Pi).
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        img_size: int = 640,
    ):
        """
        Args:
            model_path: Path ke folder model NCNN (misal 'models/best_ncnn_model').
            confidence_threshold: Threshold confidence minimum untuk deteksi.
            iou_threshold: Threshold IOU untuk Non-Maximum Suppression.
            img_size: Ukuran input gambar (sesuai imgsz saat training).
        """
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._img_size = img_size
        self._model: Any = None

    def load_model(self) -> None:
        """Muat model YOLO dari disk.

        Raises:
            ImportError: Jika ultralytics belum terinstall.
            FileNotFoundError: Jika model_path tidak ditemukan.
        """
        from ultralytics import YOLO

        logger.info(f"Loading YOLO model: {self._model_path}")
        self._model = YOLO(self._model_path)
        logger.info("YOLO model loaded successfully.")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Jalankan inference pada satu frame.

        Args:
            frame: Frame gambar (numpy array BGR dari OpenCV).

        Returns:
            List deteksi objek 'person' pada frame.
        """
        if self._model is None:
            raise RuntimeError("Model belum dimuat. Panggil load_model() terlebih dahulu.")

        results = self._model(
            frame,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            classes=0,  # Hanya kelas 'person'
            imgsz=self._img_size,
            verbose=False,
        )

        detections: list[Detection] = []
        for r in results:
            for *xyxy, conf, cls in r.boxes.data.tolist():
                if int(cls) == 0:
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(conf),
                        class_id=0,
                    ))

        return detections

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        self._confidence_threshold = value

    @property
    def iou_threshold(self) -> float:
        return self._iou_threshold

    @iou_threshold.setter
    def iou_threshold(self, value: float) -> None:
        self._iou_threshold = value

    @property
    def img_size(self) -> int:
        return self._img_size
