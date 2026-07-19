"""
app.models.config — Pydantic schemas for system configurations.
"""
from typing import Optional
from sqlmodel import Field, SQLModel

class SystemConfig(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    loitering_threshold_seconds: int = Field(default=10)
    grace_period_seconds: int = Field(default=3)
