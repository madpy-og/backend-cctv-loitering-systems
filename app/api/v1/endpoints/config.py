"""
app.api.v1.endpoints.config — System configuration endpoints.
"""
from fastapi import APIRouter, Depends
from sqlmodel import Session, select
from typing import Optional

from app.api.deps import get_behavior
from app.services.behavior import BehaviorAnalyzer
from app.models.config import SystemConfigUpdate, SystemConfigResponse, SystemConfig
from app.core.db import engine
from datetime import datetime, timezone

router = APIRouter()

@router.get("/", response_model=SystemConfigResponse)
def get_config(behavior: BehaviorAnalyzer = Depends(get_behavior)):
    """Ambil konfigurasi deteksi saat ini (threshold, grace period)."""
    return SystemConfigResponse(
        loitering_threshold_seconds=int(behavior.loitering_threshold_seconds),
        grace_period_seconds=int(behavior.grace_period_seconds)
    )

@router.put("/", response_model=SystemConfigResponse)
def update_config(
    config_in: SystemConfigUpdate, 
    behavior: BehaviorAnalyzer = Depends(get_behavior)
):
    """Update konfigurasi deteksi secara runtime."""
    # Update in memory behavior analyzer
    if config_in.loitering_threshold_seconds is not None:
        behavior.loitering_threshold_seconds = float(config_in.loitering_threshold_seconds)
    if config_in.grace_period_seconds is not None:
        behavior.grace_period_seconds = float(config_in.grace_period_seconds)
        
    # Update in Database
    with Session(engine) as session:
        statement = select(SystemConfig).limit(1)
        config_db = session.exec(statement).first()
        
        if config_db:
            if config_in.loitering_threshold_seconds is not None:
                config_db.loitering_threshold_seconds = config_in.loitering_threshold_seconds
            if config_in.grace_period_seconds is not None:
                config_db.grace_period_seconds = config_in.grace_period_seconds
            config_db.updated_at = datetime.now(timezone.utc)
            session.add(config_db)
            session.commit()
        else:
            # Create if not exists
            new_config = SystemConfig(
                loitering_threshold_seconds=int(behavior.loitering_threshold_seconds),
                grace_period_seconds=int(behavior.grace_period_seconds)
            )
            session.add(new_config)
            session.commit()
            
    return SystemConfigResponse(
        loitering_threshold_seconds=int(behavior.loitering_threshold_seconds),
        grace_period_seconds=int(behavior.grace_period_seconds)
    )
