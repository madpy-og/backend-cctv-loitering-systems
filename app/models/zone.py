"""
app.models.zone — Pydantic schemas for hazard zone CRUD operations.
"""
import json
import numpy as np
from typing import Optional, Any
from sqlmodel import Field, SQLModel
from datetime import datetime, timezone

class Zone(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    coordinates: str = Field(description="JSON string of polygon coordinates e.g. [{'x': 10, 'y': 20}, ...]")
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def points(self) -> list[dict[str, float]]:
        """Mengambil list points dari JSON string coordinates."""
        try:
            return json.loads(self.coordinates)
        except (json.JSONDecodeError, TypeError):
            return []

    def to_polygon_array(self, frame_width: int, frame_height: int) -> np.ndarray:
        """Konversi points ke numpy polygon array untuk cv2.pointPolygonTest."""
        pts = [(int(p["x"]), int(p["y"])) for p in self.points]
        if not pts:
            return np.array([[]], np.int32)
        return np.array([pts], np.int32)

class ZonePoint(SQLModel):
    x: float
    y: float

class ZoneCreate(SQLModel):
    name: str
    points: list[ZonePoint]
    is_active: bool = True

class ZoneUpdate(SQLModel):
    name: Optional[str] = None
    points: Optional[list[ZonePoint]] = None
    is_active: Optional[bool] = None

class ZoneResponse(SQLModel):
    id: int
    name: str
    is_active: bool
    created_at: datetime
    points: list[ZonePoint]
