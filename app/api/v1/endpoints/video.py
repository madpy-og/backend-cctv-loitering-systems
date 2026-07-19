"""
app.api.v1.endpoints.video — MJPEG live video stream endpoint.
"""
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.api.deps import get_frame_streamer
from app.services.frame_streamer import FrameStreamer

router = APIRouter()

@router.get("/video_feed")
def video_feed(streamer: FrameStreamer = Depends(get_frame_streamer)):
    """
    HTTP MJPEG Video stream. 
    Use this endpoint directly in an `<img>` tag in the browser.
    """
    return StreamingResponse(
        streamer.generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
