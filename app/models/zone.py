"""
app.models.zone — Pydantic schemas for hazard zone CRUD operations.
"""
from typing import Optional
from sqlmodel import Field, SQLModel
from datetime import datetime, timezone

class Zone(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    coordinates: str = Field(description="JSON string of polygon coordinates e.g. [[x1, y1], [x2, y2], ...]")
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
