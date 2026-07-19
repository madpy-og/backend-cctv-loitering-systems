"""
app.api.v1.endpoints.alerts — Endpoint for loitering alerts.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from typing import List
import os

from app.api.deps import get_alert_logger
from app.services.alert_logger import AlertLogger
from app.models.alert import Alert

router = APIRouter()

@router.get("/", response_model=List[Alert])
def get_alerts(
    limit: int = 50, 
    offset: int = 0, 
    alert_logger: AlertLogger = Depends(get_alert_logger)
):
    """Ambil daftar alert loitering terbaru (terbaru di depan)."""
    return alert_logger.get_alerts(limit=limit, offset=offset)

@router.get("/{alert_id}/snapshot")
def get_alert_snapshot(
    alert_id: int, 
    alert_logger: AlertLogger = Depends(get_alert_logger)
):
    """Ambil gambar snapshot untuk alert tertentu."""
    alert = alert_logger.get_alert_by_id(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert tidak ditemukan")
    
    full_path = alert_logger.get_snapshot_path(alert.snapshot_path)
    if not full_path:
        raise HTTPException(status_code=404, detail="Snapshot tidak ditemukan")
        
    return FileResponse(str(full_path), media_type="image/jpeg")
