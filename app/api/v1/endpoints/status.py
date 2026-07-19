from fastapi import APIRouter, Depends
from app.core.config import settings
from app.api.deps import get_pipeline, get_zone_manager
from app.services.pipeline import PipelineService
from app.services.zone_manager import ZoneManager

router = APIRouter()

@router.get("/", summary="Check API health/status")
def get_status(
    pipeline: PipelineService = Depends(get_pipeline),
    zone_manager: ZoneManager = Depends(get_zone_manager)
):
    zones = zone_manager.list_zones()
    active_zones_count = sum(1 for z in zones if z.is_active)
    
    return {
        "status": "healthy",
        "project_name": settings.PROJECT_NAME,
        "api_version": "1.0.0",
        "debug_mode": settings.DEBUG,
        "pipeline_status": pipeline.status,
        "pipeline_fps": pipeline.current_fps,
        "active_zones_count": active_zones_count
    }
