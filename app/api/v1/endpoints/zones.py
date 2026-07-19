"""
app.api.v1.endpoints.zones — CRUD endpoints for hazard zones.
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import List

from app.api.deps import get_zone_manager
from app.services.zone_manager import ZoneManager
from app.models.zone import ZoneCreate, ZoneUpdate, ZoneResponse, Zone

router = APIRouter()

@router.get("/", response_model=List[ZoneResponse])
def get_zones(zone_manager: ZoneManager = Depends(get_zone_manager)):
    """Ambil daftar semua zona yang terdaftar."""
    zones = zone_manager.list_zones()
    return [
        ZoneResponse(
            id=z.id,  # type: ignore
            name=z.name, 
            is_active=z.is_active, 
            created_at=z.created_at, 
            points=[p for p in z.points]
        ) for z in zones
    ]

@router.post("/", response_model=ZoneResponse)
def create_zone(zone_in: ZoneCreate, zone_manager: ZoneManager = Depends(get_zone_manager)):
    """Buat zona bahaya baru."""
    points_dict = [p.model_dump() for p in zone_in.points]
    try:
        z = zone_manager.create_zone(name=zone_in.name, points=points_dict, is_active=zone_in.is_active)
        return ZoneResponse(
            id=z.id,  # type: ignore
            name=z.name, 
            is_active=z.is_active, 
            created_at=z.created_at, 
            points=[p for p in z.points]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/{zone_id}", response_model=ZoneResponse)
def update_zone(zone_id: int, zone_in: ZoneUpdate, zone_manager: ZoneManager = Depends(get_zone_manager)):
    """Update konfigurasi zona."""
    points_dict = [p.model_dump() for p in zone_in.points] if zone_in.points is not None else None
    
    try:
        z = zone_manager.update_zone(
            zone_id=zone_id, 
            name=zone_in.name, 
            points=points_dict, 
            is_active=zone_in.is_active
        )
        if not z:
            raise HTTPException(status_code=404, detail="Zona tidak ditemukan")
        return ZoneResponse(
            id=z.id,  # type: ignore
            name=z.name, 
            is_active=z.is_active, 
            created_at=z.created_at, 
            points=[p for p in z.points]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{zone_id}")
def delete_zone(zone_id: int, zone_manager: ZoneManager = Depends(get_zone_manager)):
    """Hapus zona berdasarkan ID."""
    success = zone_manager.delete_zone(zone_id)
    if not success:
        raise HTTPException(status_code=404, detail="Zona tidak ditemukan")
    return {"message": "Zona berhasil dihapus"}
