from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.v1.router import api_router
from app.core.db import init_db

# Import models so SQLModel metadata can register them
from app.models.zone import Zone
from app.models.alert import Alert
from app.models.config import SystemConfig

from app.services.camera import CameraService
from app.services.detector import DetectorService
from app.services.tracker import TrackerService
from app.services.behavior import BehaviorAnalyzer
from app.services.zone_manager import ZoneManager
from app.services.alert_logger import AlertLogger
from app.services.frame_streamer import FrameStreamer
from app.services.pipeline import PipelineService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the database and create tables
    init_db()

    # Initialize Services
    camera = CameraService(source=settings.VIDEO_SOURCE)
    detector = DetectorService(
        model_path=settings.MODEL_PATH,
        confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        img_size=settings.IMGSZ
    )
    tracker = TrackerService()
    zone_manager = ZoneManager()
    zone_manager.load()
    behavior = BehaviorAnalyzer(
        zone_manager=zone_manager,
        loitering_threshold_seconds=settings.LOITERING_THRESHOLD_SECONDS,
        grace_period_seconds=settings.GRACE_PERIOD_SECONDS
    )
    alert_logger = AlertLogger(snapshots_dir=settings.SNAPSHOTS_DIR)
    frame_streamer = FrameStreamer()

    pipeline = PipelineService(
        camera=camera,
        detector=detector,
        tracker=tracker,
        behavior=behavior,
        zone_manager=zone_manager,
        alert_logger=alert_logger,
        frame_streamer=frame_streamer,
        frame_skip=settings.FRAME_SKIP
    )

    # Attach to app state for dependency injection
    app.state.zone_manager = zone_manager
    app.state.alert_logger = alert_logger
    app.state.pipeline = pipeline
    app.state.behavior = behavior
    app.state.frame_streamer = frame_streamer

    # Start pipeline
    pipeline.start()
    
    yield
    
    # Cleanup logic
    pipeline.stop()
    camera.release()

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

# Mount static files for snapshots
app.mount("/static/snapshots", StaticFiles(directory=settings.SNAPSHOTS_DIR), name="snapshots")

@app.get("/")
def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs_url": f"{settings.API_V1_STR}/docs"
    }
