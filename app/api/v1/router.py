from fastapi import APIRouter
from app.api.v1.endpoints import status, video, zones, alerts, config

api_router = APIRouter()
api_router.include_router(status.router, prefix="/status", tags=["status"])
api_router.include_router(video.router, tags=["video"])
api_router.include_router(zones.router, prefix="/zones", tags=["zones"])
api_router.include_router(alerts.router, prefix="/alerts", tags=["alerts"])
api_router.include_router(config.router, prefix="/config", tags=["config"])
