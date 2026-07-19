from sqlmodel import SQLModel, create_engine, Session
from typing import Generator
from app.core.config import settings
import os

# Ensure the data directory exists
os.makedirs(os.path.dirname(settings.DATABASE_URL.replace("sqlite:///", "")), exist_ok=True)

# Create sqlite engine. connect_args is needed for SQLite to avoid thread-related errors
# in concurrent scenarios since FastAPI is multi-threaded.
connect_args = {"check_same_thread": False}
engine = create_engine(settings.DATABASE_URL, echo=settings.DEBUG, connect_args=connect_args)

def init_db() -> None:
    # This will create tables for all models that inherit from SQLModel
    SQLModel.metadata.create_all(engine)

def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
