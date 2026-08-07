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

    embedding_model: str = "openai/text-embedding-3-small"

    # paper
    read_max_papers: int = 5

    # chunk
    fulltext_chunk_chars: int = 1500
    fulltext_chunk_overlap: int = 200
    fulltext_top_k: int = 5
    fulltext_excerpt_max_chars: int = 4000

    arxiv_max_results: int = 8
    openalex_max_results: int = 8
    # HTTP calls to paper providers -- short, they either answer or they don't.
    request_timeout_seconds: int = 20
    max_new_papers_per_round: int = 25

    # LLM calls are a different beast: a reasoning-tier gap analysis emitting
    # verbatim quotes routinely runs past a provider HTTP timeout, and every
    # timeout costs 3 attempts via invoke_with_retry before surfacing.
    llm_timeout_seconds: int = Field(default=180, alias="LLM_TIMEOUT_SECONDS")

    # cross-run memory: full-text index + prior-gap recall. Off by default so a
    # plain run stays ephemeral; the TUI turns it on.
    use_memory: bool = Field(default=False, alias="USE_MEMORY")
    memory_dir: str = Field(default=".paper_research_memory", alias="MEMORY_DIR")
    result_cache_ttl_days: int = Field(default=7, alias="RESULT_CACHE_TTL_DAYS")


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
