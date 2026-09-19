"""Application settings, loaded from environment variables and an optional ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JEVFLOW_", env_file=".env", extra="ignore")

    # TypeSafe Jev -------------------------------------------------------------
    typesafe_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("TYPESAFE_API_KEY", "JEVFLOW_TYPESAFE_API_KEY")
    )
    jev_model: str = "jev-latest"
    jev_timeout_s: float = Field(3.0, gt=0, description="Per-request budget; control loops cannot wait long")
    jev_max_retries: int = Field(1, ge=0)
    jev_decision_interval_s: float = Field(1.0, ge=0.5)
    jev_ask_congestion: bool = Field(True, description="Also ask Jev to score congestion (for the dashboard)")

    # Runtime -----------------------------------------------------------------
    database_path: Path = Path("data/jevflow.db")
    max_live_sessions: int = Field(4, ge=1)
    stream_fps: float = Field(15.0, gt=0, le=60)
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_url: str = "http://127.0.0.1:8000"
    public_ws_url: str | None = Field(None, description="WebSocket base URL as seen by the browser")

    @property
    def jev_enabled(self) -> bool:
        return self.typesafe_api_key is not None and bool(self.typesafe_api_key.get_secret_value().strip())

    @property
    def ws_url(self) -> str:
        return self.public_ws_url or self.api_url.replace("http", "ws", 1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
