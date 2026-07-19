"""
app.models.alert — Pydantic schemas for loitering alert responses.
"""
from typing import Optional
from sqlmodel import Field, SQLModel
from datetime import datetime, timezone

class Alert(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), index=True)
    track_id: int
    duration: float
    snapshot_path: Optional[str] = None
    zone_id: Optional[int] = Field(default=None, foreign_key="zone.id")
    is_resolved: bool = Field(default=False)
