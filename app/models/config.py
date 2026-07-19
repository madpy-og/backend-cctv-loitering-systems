"""
app.models.config — Pydantic schemas for system configurations.
"""
from typing import Optional
from sqlmodel import Field, SQLModel
from datetime import datetime, timezone

class SystemConfig(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    loitering_threshold_seconds: int = Field(default=10)
    grace_period_seconds: int = Field(default=3)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SystemConfigUpdate(SQLModel):
    loitering_threshold_seconds: Optional[int] = None
    grace_period_seconds: Optional[int] = None

class SystemConfigResponse(SQLModel):
    loitering_threshold_seconds: int
    grace_period_seconds: int
