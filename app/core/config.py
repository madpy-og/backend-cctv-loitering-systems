from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Backend"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True
    DATABASE_URL: str = "sqlite:///./data/cctv.db"
    
    # Camera / Video source
    VIDEO_SOURCE: str = "0"  # "0" untuk webcam, atau path ke file video

    # Model
    MODEL_PATH: str = "models/best_ncnn_model"

    # Detection parameters
    LOITERING_THRESHOLD_SECONDS: float = 30.0
    GRACE_PERIOD_SECONDS: float = 5.0
    CONFIDENCE_THRESHOLD: float = 0.5

    # Storage paths
    SNAPSHOTS_DIR: str = "data/snapshots"
    ALERTS_LOG_PATH: str = "data/logs/alerts.csv"
    ZONES_CONFIG_PATH: str = "data/zones.json"

    # Pipeline
    FRAME_SKIP: int = 0
    IMGSZ: int = 640

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
