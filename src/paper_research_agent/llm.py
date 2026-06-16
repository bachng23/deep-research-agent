from __future__ import annotations

import logging
from functools import lru_cache

from langchain_openai import ChatOpenAI

from paper_research_agent.config import ModelTier, get_settings, model_for_tier

logger = logging.getLogger(__name__)


@lru_cache
def chat_model_for_tier(tier: ModelTier, temperature: float = 0.0) -> ChatOpenAI:
    settings = get_settings()

    return ChatOpenAI(
        model=model_for_tier(tier),
        api_key=settings.api_key,
        base_url=settings.llm_base_url,
        temperature=temperature,
        timeout=settings.request_timeout_seconds,
    )


def invoke_with_retry(structured_model, messages, *, retries: int = 2):
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            return structured_model.invoke(messages)
        except Exception as e:
            last_error = e
            logger.warning(
                "structured ouput failed (attempt %d/%d): %s",
                attempt + 1,
                retries + 1,
                e,
            )

    assert last_error is not None
    raise last_error
