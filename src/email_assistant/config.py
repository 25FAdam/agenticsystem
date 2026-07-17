"""Application settings loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ASSISTANT_",
        extra="ignore",
    )

    # ANTHROPIC_API_KEY is read without the ASSISTANT_ prefix so the standard
    # variable name keeps working.
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")

    model: str = "claude-opus-4-8"
    # Optional override for the high-volume triage classifier (e.g. claude-haiku-4-5).
    triage_model: str = ""

    user_name: str = "Adam Farkas"
    user_email: str = "adam.farkas@acme.io"

    data_dir: Path = Path("data")
    log_level: str = "INFO"

    @property
    def fixtures_path(self) -> Path:
        return self.data_dir / "fixtures" / "inbox.json"

    @property
    def memory_path(self) -> Path:
        return self.data_dir / "memory" / "memory.json"

    @property
    def memory_seed_path(self) -> Path:
        return self.data_dir / "memory" / "memory.seed.json"

    @property
    def actions_path(self) -> Path:
        return self.data_dir / "actions.json"

    @property
    def calendar_path(self) -> Path:
        return self.data_dir / "calendar.json"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def effective_triage_model(self) -> str:
        return self.triage_model or self.model


@lru_cache
def get_settings() -> Settings:
    return Settings()
