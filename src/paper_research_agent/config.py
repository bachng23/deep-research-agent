from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ModelTier = Literal["fast", "balanced", "reasoning"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    api_key: str = Field(..., alias="API_KEY")
    llm_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="LLM_BASE_URL"
    )

    openalex_api_key: str | None = Field(default=None, alias="OPENALEX_API_KEY")

    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="paper-research-agent", alias="LANGSMITH_PROJECT"
    )

    model_tier_override: ModelTier | None = Field(
        default=None, alias="MODEL_TIER_OVERRIDE"
    )

    fast_model: str = "deepseek/deepseek-v4-flash"
    balanced_model: str = "deepseek/deepseek-v4-pro"
    reasoning_model: str = "deepseek/deepseek-v4-pro"

    arxiv_max_results: int = 8
    openalex_max_results: int = 8
    request_timeout_seconds: int = 20
    max_new_papers_per_round: int = 25


@lru_cache
def get_settings() -> Settings:
    return Settings()


def model_for_tier(tier: ModelTier) -> str:
    settings = get_settings()

    effective_tier = settings.model_tier_override or tier

    if effective_tier == "fast":
        return settings.fast_model

    if effective_tier == "balanced":
        return settings.balanced_model

    return settings.reasoning_model
