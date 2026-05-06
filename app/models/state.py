"""LangGraph 共享状态模型。"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage


class ParsedRequest(TypedDict, total=False):
    """把自然语言请求解析成可执行字段。

    这些字段是后续工具调用的“硬前置条件”。
    如果缺字段，`ensure_requirements` 会 interrupt 暂停并向用户提问。
    """

    origin: str
    destination: str
    departure_date: str
    return_date: str
    preferences: str


class PlanTask(TypedDict):
    """父 Agent 拆给子 Agent 的一条任务。

    任务里同时包含：
    - display_name/goal：用于人类可读的执行轨迹
    - service/tool_name/arguments：用于实际 MCP 工具调用
    """

    section: str
    display_name: str
    service: str
    tool_name: str
    goal: str
    arguments: dict[str, Any]


class SectionResult(TypedDict):
    """子 Agent 执行后的结果。

    - content：最终用于攻略分区展示的文本
    - raw_result：结构化工具结果（便于调试、复用、再渲染）
    - source：cache 或 mcp，帮助理解缓存命中情况
    """

    section: str
    content: str
    source: str
    raw_result: dict[str, Any]


class TravelAssistantState(TypedDict, total=False):
    """整个图在节点之间共享的状态。

    这个 state 是 LangGraph 节点之间传递的“共享内存”。核心字段：
    - messages：包含 Human/AI/ToolMessage，会在节点边界被压缩
    - plan：父 Agent 规划出的任务
    - agent_outputs：子 Agent 执行结果（按 section 键控）
    - guide_sections/final_guide：Replan 后的最终攻略
    """

    session_id: str
    mode: Literal["initial", "feedback"]
    request_text: str
    parsed_request: ParsedRequest
    messages: list[BaseMessage]
    missing_fields: list[str]
    pending_question: str
    feedback_text: str
    affected_sections: list[str]
    plan: list[PlanTask]
    agent_outputs: dict[str, SectionResult]
    guide_sections: dict[str, str]
    final_guide: str
    status: str

