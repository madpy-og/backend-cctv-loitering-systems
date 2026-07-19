from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Backend"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True
    DATABASE_URL: str = "sqlite:///./data/cctv.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
