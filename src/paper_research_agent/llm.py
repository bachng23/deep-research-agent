from __future__ import annotations

import logging
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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
        timeout=settings.llm_timeout_seconds,
        # invoke_with_retry already retries; letting the client retry too
        # multiplies the worst case instead of bounding it.
        max_retries=0,
    )


def invoke_with_retry(structured_model, messages, *, retries: int = 2, is_valid=None):
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            result = structured_model.invoke(messages)
        except Exception as e:
            last_error = e
            logger.warning(
                "structured output failed (attempt %d/%d): %s",
                attempt + 1,
                retries + 1,
                e,
            )
            continue
        if is_valid is not None and not is_valid(result):
            last_error = ValueError("structured output failed validation (empty?)")
            logger.warning(
                "structured output invalid (attempt %d/%d)", attempt + 1, retries + 1
            )
            continue
        return result

    assert last_error is not None
    raise last_error


@lru_cache
def embeddings_model() -> OpenAIEmbeddings:
    settings = get_settings()

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.api_key,
        base_url=settings.llm_base_url,
        # Without this the client waits forever: an embedding call that stops
        # responding mid-stream stalled a 39-run eval for 68 minutes with the
        # socket still open.
        timeout=settings.llm_timeout_seconds,
        max_retries=2,
    )
