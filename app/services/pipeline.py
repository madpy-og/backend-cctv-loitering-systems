"""
app.services.pipeline — Orchestrator: menjalankan pipeline deteksi di thread terpisah.

Refactored from: loitering_system.py → main() function (lines 165-319).

Loop per-frame:
  camera.read() → detector.detect() → tracker.update() →
  behavior.analyze() → frame_streamer.update() → alert_logger.log()

NFR: Proses deteksi berjalan sebagai thread terpisah dari web server,
sehingga kegagalan salah satu sisi tidak langsung menjatuhkan yang lain.
"""

import time
import threading
import logging
from enum import Enum

from app.services.camera import CameraService
from app.services.detector import DetectorService
from app.services.tracker import TrackerService
from app.services.behavior import BehaviorAnalyzer
from app.services.zone_manager import ZoneManager
from app.services.alert_logger import AlertLogger
from app.services.frame_streamer import FrameStreamer

logger = logging.getLogger(__name__)


class PipelineStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


class PipelineService:
    """Orchestrator utama: menjalankan pipeline detection + tracking + behavior
    analysis dalam thread terpisah.

    Lifecycle: start() → (loop berjalan) → stop()
    """

    def __init__(
        self,
        camera: CameraService,
        detector: DetectorService,
        tracker: TrackerService,
        behavior: BehaviorAnalyzer,
        zone_manager: ZoneManager,
        alert_logger: AlertLogger,
        frame_streamer: FrameStreamer,
        frame_skip: int = 0,
    ):
        self._camera = camera
        self._detector = detector
        self._tracker = tracker
        self._behavior = behavior
        self._zone_manager = zone_manager
        self._alert_logger = alert_logger
        self._frame_streamer = frame_streamer
        self._frame_skip = frame_skip

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._status = PipelineStatus.STOPPED
        self._current_fps: float = 0.0
        self._frame_count: int = 0
        self._error_message: str = ""

    def start(self) -> bool:
        """Start pipeline dalam thread terpisah.

        Returns:
            True jika berhasil dimulai, False jika sudah berjalan.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Pipeline already running.")
            return False

        self._stop_event.clear()
        self._status = PipelineStatus.STARTING
        self._error_message = ""
        self._frame_count = 0

        self._thread = threading.Thread(target=self._run_loop, name="pipeline-thread", daemon=True)
        self._thread.start()

        logger.info("Pipeline thread started.")
        return True

    def stop(self) -> None:
        """Stop pipeline (blocking sampai thread selesai)."""
        if self._thread is None:
            return

        logger.info("Stopping pipeline...")
        self._stop_event.set()
        self._thread.join(timeout=10.0)
        self._thread = None
        self._status = PipelineStatus.STOPPED
        self._current_fps = 0.0
        logger.info("Pipeline stopped.")

    def _run_loop(self) -> None:
        """Main processing loop — berjalan di thread terpisah."""
        try:
            # --- Inisialisasi ---
            if not self._camera.open():
                raise RuntimeError(f"Gagal membuka video source: {self._camera.source}")

            self._detector.load_model()
            self._tracker.initialize()

            fps = self._camera.fps or 30.0  # Fallback 30 fps jika unknown (webcam)
            self._status = PipelineStatus.RUNNING
            logger.info(f"Pipeline running. Source FPS: {fps:.1f}, frame_skip: {self._frame_skip}")

            frame_count = 0
            skip_counter = 0
            fps_timer_start = time.time()
            fps_frame_count = 0

            while not self._stop_event.is_set():
                frame = self._camera.read_frame()
                if frame is None:
                    # Video habis (file) atau kamera disconnect
                    logger.info("No more frames available. Ending pipeline loop.")
                    break

                frame_count += 1

                # Frame skipping
                if self._frame_skip > 0:
                    skip_counter += 1
                    if skip_counter <= self._frame_skip:
                        continue
                    skip_counter = 0

                # --- Detection ---
                detections = self._detector.detect(frame)

                # --- Tracking ---
                tracks = self._tracker.update(detections, frame)

                frame_h, frame_w = frame.shape[:2]

                # --- Behavior Analysis ---
                alerts = self._behavior.analyze(
                    tracks=tracks,
                    current_frame=frame_count,
                    fps=fps,
                    frame_width=frame_w,
                    frame_height=frame_h,
                )

                # --- Frame overlay + buffer update ---
                output_frame = self._frame_streamer.update_frame(
                    frame=frame,
                    tracks=tracks,
                    behavior_analyzer=self._behavior,
                    zone_manager=self._zone_manager,
                    current_frame=frame_count,
                    fps=fps,
                )

                # --- Log alerts + save snapshots ---
                for alert in alerts:
                    self._alert_logger.log_alert(event=alert, frame=output_frame)

                # --- FPS calculation ---
                fps_frame_count += 1
                elapsed = time.time() - fps_timer_start
                if elapsed >= 1.0:
                    self._current_fps = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_timer_start = time.time()

                self._frame_count = frame_count

        except Exception as e:
            self._status = PipelineStatus.ERROR
            self._error_message = str(e)
            logger.exception(f"Pipeline error: {e}")
        finally:
            self._camera.release()
            if self._status != PipelineStatus.ERROR:
                self._status = PipelineStatus.STOPPED
            logger.info(f"Pipeline loop finished. Total frames processed: {self._frame_count}")

    # --- Status getters ---

    @property
    def status(self) -> PipelineStatus:
        return self._status

    @property
    def current_fps(self) -> float:
        return round(self._current_fps, 1)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def error_message(self) -> str:
        return self._error_message

    @property
    def is_running(self) -> bool:
        return self._status == PipelineStatus.RUNNING
