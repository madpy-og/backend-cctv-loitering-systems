"""
app.models.zone — Pydantic schemas for hazard zone CRUD operations.
"""
from typing import Optional
from sqlmodel import Field, SQLModel

class Zone(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    coordinates: str = Field(description="JSON string of polygon coordinates e.g. [[x1, y1], [x2, y2], ...]")
