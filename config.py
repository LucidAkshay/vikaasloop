# config.py
# AGPL v3 - VikaasLoop

import os
import secrets
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    GEMINI_API_KEY: str = Field(default="")
    BASE_MODEL_NAME: str = Field(default="microsoft/phi-2")
    FINETUNED_MODEL_PATH: str = Field(default="models/finetuned")
    ADAPTER_PATH: str = Field(default="models/adapter")
    DB_PATH: str = Field(default="data/eval_results.db")
    SKILLS_DB_PATH: str = Field(default="data/skills.db")
    TEST_PROMPTS_PATH: str = Field(default="data/test_prompts.jsonl")

    JWT_SECRET: Optional[str] = Field(default=None)
    ENV: str = Field(default="development")

    _generated_secret: Optional[str] = None

    @property
    def get_jwt_secret(self) -> str:
        """Returns the configured JWT_SECRET or a session consistent random one if in dev."""
        if self.JWT_SECRET:
            return self.JWT_SECRET

        if self.ENV == "production":
            raise RuntimeError(
                "CRITICAL: JWT_SECRET must be set in production environment."
            )

        if self._generated_secret is None:
            self._generated_secret = secrets.token_hex(32)

        return self._generated_secret

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
