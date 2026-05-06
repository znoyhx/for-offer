"""FastAPI 请求与响应模型（Pydantic）。

这些模型是 API 对外的“稳定协议”：
- 输入：自然语言 request / 补充 reply / 局部反馈 feedback
- 输出：统一的 TravelResponse

注意：即使内部工作流更复杂（LangGraph + 多 Agent + interrupt），
对外仍然保持最小、可理解的 3 个交互动作。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlanRequest(BaseModel):
    """创建旅游规划的入口请求。"""

    request: str = Field(..., min_length=1, description="用户的自然语言旅游需求")


class ResumeRequest(BaseModel):
    """补充缺失信息时使用。"""

    reply: str = Field(..., min_length=1, description="用户对补充问题的回答")


class FeedbackRequest(BaseModel):
    """对已有攻略提出局部修改意见。"""

    feedback: str = Field(..., min_length=1, description="用户反馈，例如“住宿便宜一点”")


class TravelResponse(BaseModel):
    """统一响应结构。"""

    session_id: str
    status: Literal["needs_input", "completed"]
    question: str | None = None
    guide: str | None = None

