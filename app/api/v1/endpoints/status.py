from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()

@router.get("/status", summary="Check API health/status")
def get_status():
    return {
        "status": "healthy",
        "project_name": settings.PROJECT_NAME,
        "api_version": "1.0.0",
        "debug_mode": settings.DEBUG
    }
