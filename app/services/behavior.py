"""
app.services.behavior — Loitering behavior analysis.

Logika inti: zona bahaya + dwell time akumulatif + grace period.
Refactored from: loitering_system.py lines 228-291.

Mendukung multiple zona bahaya (original hanya 1 hardcoded polygon).
"""

import logging
from dataclasses import dataclass, field

from app.services.tracker import Track
from app.services.zone_manager import Zone, ZoneManager
from app.utils.geometry import point_in_polygon, get_centroid

logger = logging.getLogger(__name__)


@dataclass
class LoiteringState:
    """State loitering per track_id per zone_id."""
    in_zone: bool = False
    start_frame: int = -1
    last_in_zone_frame: int = -1
    alert_triggered: bool = False
    is_loitering: bool = False
    last_centroid: tuple[float, float] = (0.0, 0.0)


@dataclass
class AlertEvent:
    """Event alert loitering yang baru terpicu."""
    track_id: int
    zone_id: int
    zone_name: str
    centroid: tuple[float, float]
    dwell_time_seconds: float
    frame_number: int


class BehaviorAnalyzer:
    """Menganalisis perilaku loitering berdasarkan zona bahaya dan durasi.

    Fitur utama (sesuai PRD FR-2.1 s/d FR-2.5):
    - Cek centroid tiap track di dalam polygon zona bahaya aktif.
    - Hitung dwell time akumulatif per track per zona.
    - Terapkan grace period (toleransi keluar sesaat akibat noise tracking).
    - Trigger alert SEKALI per sesi loitering (tidak berulang selama sesi sama).
    """

    def __init__(
        self,
        zone_manager: ZoneManager,
        loitering_threshold_seconds: float = 30.0,
        grace_period_seconds: float = 5.0,
    ):
        self._zone_manager = zone_manager
        self._loitering_threshold_seconds = loitering_threshold_seconds
        self._grace_period_seconds = grace_period_seconds

        # State per (track_id, zone_id)
        self._state: dict[tuple[int, int], LoiteringState] = {}

    @property
    def loitering_threshold_seconds(self) -> float:
        return self._loitering_threshold_seconds

    @loitering_threshold_seconds.setter
    def loitering_threshold_seconds(self, value: float) -> None:
        self._loitering_threshold_seconds = value

    @property
    def grace_period_seconds(self) -> float:
        return self._grace_period_seconds

    @grace_period_seconds.setter
    def grace_period_seconds(self, value: float) -> None:
        self._grace_period_seconds = value

    def analyze(
        self,
        tracks: list[Track],
        current_frame: int,
        fps: float,
        frame_width: int,
        frame_height: int,
    ) -> list[AlertEvent]:
        """Analisis perilaku loitering pada frame saat ini.

        Args:
            tracks: List track dari TrackerService (hanya yang confirmed).
            current_frame: Nomor frame saat ini.
            fps: FPS video (untuk konversi frame ke detik).
            frame_width: Lebar frame (untuk konversi koordinat zona).
            frame_height: Tinggi frame (untuk konversi koordinat zona).

        Returns:
            List AlertEvent untuk alert yang baru terpicu pada frame ini.
        """
        active_zones = self._zone_manager.get_active_zones()
        if not active_zones:
            return []

        alerts: list[AlertEvent] = []

        for track in tracks:
            centroid = get_centroid(track.bbox)

            for zone in active_zones:
                polygon = zone.to_polygon_array(frame_width, frame_height)
                
                if zone.id is None:
                    continue

                state_key = (track.track_id, zone.id)

                if state_key not in self._state:
                    self._state[state_key] = LoiteringState(last_centroid=centroid)

                state = self._state[state_key]
                is_in_zone = point_in_polygon(centroid, polygon)

                if is_in_zone:
                    # Masuk atau masih di dalam zona
                    if state.start_frame == -1:
                        state.start_frame = current_frame
                        state.alert_triggered = False
                        state.is_loitering = False

                    state.in_zone = True
                    state.last_in_zone_frame = current_frame
                    state.last_centroid = centroid

                    # Hitung elapsed time
                    elapsed_frames = current_frame - state.start_frame
                    elapsed_time_sec = elapsed_frames / fps if fps > 0 else 0

                    # Cek apakah sudah melewati threshold
                    if (elapsed_time_sec >= self._loitering_threshold_seconds
                            and not state.alert_triggered):
                        state.is_loitering = True
                        state.alert_triggered = True

                        alert = AlertEvent(
                            track_id=track.track_id,
                            zone_id=zone.id,
                            zone_name=zone.name,
                            centroid=centroid,
                            dwell_time_seconds=elapsed_time_sec,
                            frame_number=current_frame,
                        )
                        alerts.append(alert)

                        logger.info(
                            f"[ALERT] Track {track.track_id} loitering di '{zone.name}' "
                            f"selama {elapsed_time_sec:.2f}s pada frame {current_frame}"
                        )
                else:
                    # Di luar zona
                    state.in_zone = False
                    if state.start_frame != -1:
                        frames_since_last = current_frame - state.last_in_zone_frame
                        grace_frames = self._grace_period_seconds * fps if fps > 0 else 0
                        if frames_since_last > grace_frames:
                            # Grace period habis, reset state
                            state.start_frame = -1
                            state.last_in_zone_frame = -1
                            state.is_loitering = False
                            state.alert_triggered = False

        return alerts

    def get_track_state(self, track_id: int, zone_id: int) -> LoiteringState | None:
        """Ambil state loitering untuk track + zona tertentu."""
        return self._state.get((track_id, zone_id))

    def get_all_states(self) -> dict[tuple[int, int], LoiteringState]:
        """Ambil semua state (untuk overlay visual)."""
        return self._state.copy()

    def reset(self) -> None:
        """Reset semua state loitering."""
        self._state.clear()
        logger.info("Behavior analyzer state reset.")
