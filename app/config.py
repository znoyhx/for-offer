"""项目配置。"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """集中管理这个学习项目用到的可调参数。"""

    app_name: str = "AI Travel Assistant Learning Demo"
    # 在 `compress_messages()` 中保留的“最近非 ToolMessage 条数”。
    # 值越大，上下文越完整，但 token 成本越高。
    recent_message_limit: int = 6

    # 上下文 token 上限（近似值）。节点边界会裁剪消息，避免无限增长。
    max_context_tokens: int = 1200

    # 缓存相似命中阈值（Jaccard token overlap）。越高越保守，越低越容易复用。
    cache_similarity_threshold: float = 0.88

    # 子 Agent 每个任务最多尝试几轮“查缓存/调工具/检查完整性”。
    # 对应 worker.py 的 ReAct 风格循环。
    max_tool_attempts: int = 2

    model_config = SettingsConfigDict(
        env_prefix="TRAVEL_ASSISTANT_",
        extra="ignore",
    )


settings = Settings()

