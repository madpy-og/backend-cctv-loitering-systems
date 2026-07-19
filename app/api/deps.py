"""
app.api.deps — Dependency injection for FastAPI endpoints.
Provides access to shared service instances.
"""
from fastapi import Request

from app.services.zone_manager import ZoneManager
from app.services.alert_logger import AlertLogger
from app.services.pipeline import PipelineService
from app.services.behavior import BehaviorAnalyzer
from app.services.frame_streamer import FrameStreamer

def get_zone_manager(request: Request) -> ZoneManager:
    return request.app.state.zone_manager

def get_alert_logger(request: Request) -> AlertLogger:
    return request.app.state.alert_logger

def get_pipeline(request: Request) -> PipelineService:
    return request.app.state.pipeline

def get_behavior(request: Request) -> BehaviorAnalyzer:
    return request.app.state.behavior

def get_frame_streamer(request: Request) -> FrameStreamer:
    return request.app.state.frame_streamer
