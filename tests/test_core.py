"""核心工具测试。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.cache import ToolResultCache
from app.core.context import compress_messages


def test_cache_can_reuse_same_query() -> None:
    cache = ToolResultCache(similarity_threshold=0.8)
    result = {"options": [{"name": "样例酒店"}], "advice": "示例建议"}

    cache.store(
        "hotel",
        "search_hotels",
        {"destination": "上海", "departure_date": "2026-05-01"},
        result,
    )

    hit = cache.lookup(
        "hotel",
        "search_hotels",
        {"destination": "上海", "departure_date": "2026-05-01"},
    )

    assert hit == result
    assert hit is not result


def test_context_compression_keeps_tool_messages_and_recent_dialogue() -> None:
    messages = [
        HumanMessage(content="u1"),
        AIMessage(content="a1"),
        HumanMessage(content="u2"),
        ToolMessage(content="tool-result", tool_call_id="tool-1"),
        HumanMessage(content="u3"),
        AIMessage(content="a3"),
        HumanMessage(content="u4"),
    ]

    compressed = compress_messages(messages, recent_non_tool=2, max_tokens=500)

    assert any(isinstance(message, ToolMessage) for message in compressed)
    assert compressed[-1].content == "u4"
    assert sum(not isinstance(message, ToolMessage) for message in compressed) <= 2

