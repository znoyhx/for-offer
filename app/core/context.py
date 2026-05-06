"""上下文压缩工具。"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, ToolMessage, trim_messages


def compress_messages(
    messages: list[BaseMessage],
    *,
    recent_non_tool: int,
    max_tokens: int,
) -> list[BaseMessage]:
    """保留所有 ToolMessage + 最近 N 条普通消息。

    这正对应草稿里的“保留 ToolMessage + 最近 N 条消息”的要求。
    最后再用 LangChain 的 `trim_messages` 做一次近似 token 限流。
    """

    # 空输入直接返回，方便节点代码保持简洁。
    if not messages:
        return []

    indexed_messages = list(enumerate(messages))
    # 规则 1：保留所有 ToolMessage。
    # 对工具型 Agent 来说，工具返回值往往比闲聊更“可复用/可解释”。
    kept_indices = {index for index, message in indexed_messages if isinstance(message, ToolMessage)}
    non_tool_indices = [index for index, message in indexed_messages if not isinstance(message, ToolMessage)]

    # 规则 2：保留最近 N 条非 ToolMessage，让用户意图与最近对话不断线。
    kept_indices.update(non_tool_indices[-recent_non_tool:])
    selected_messages = [messages[index] for index in sorted(kept_indices)]

    # 规则 3：再做一次“近似 token 裁剪”。
    # 这是最后一道保险，避免图执行过程中消息无限增长。
    return trim_messages(
        selected_messages,
        max_tokens=max_tokens,
        token_counter="approximate",
        strategy="last",
        include_system=True,
    )

