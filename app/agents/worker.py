"""执行 MCP 工具调用的专业子 Agent（子 Agent 执行器）。

在这个学习项目里，“多 Agent”不是指 6 个大模型互聊，而是：

- 父 Agent（`LearningLLM.build_plan`）把需求拆成多个领域任务
- 子 Agent 执行器（本文件）用统一流程执行每个任务：
    1) 思考：本轮要做什么
    2) 行动：先查缓存，未命中再调用 MCP 工具
    3) 观察：检查结构化结果是否满足完成标准
    4) 结束/继续：不完整就进入下一轮（ReAct 风格循环）

这样写的好处是：
- 缓存复用成为执行路径的一部分（不是事后优化）
- 每类任务有清晰的“完成标准”，可解释、可测试
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app.core.cache import ToolResultCache
from app.core.llm import LearningLLM
from app.mcp.client import McpClientManager
from app.models.state import PlanTask, SectionResult


class TravelWorkerAgent:
    """统一的子 Agent 执行器。

    不做“真多智能体对话”，而是保留一层清晰的 ReAct 结构：
    思考 -> 调工具/查缓存 -> 观察 -> 判断是否结束
    """

    def __init__(
        self,
        *,
        cache: ToolResultCache,
        mcp_client: McpClientManager,
        llm: LearningLLM,
        max_tool_attempts: int,
    ) -> None:
        self._cache = cache
        self._mcp_client = mcp_client
        self._llm = llm
        self._max_tool_attempts = max_tool_attempts

    async def run(self, task: PlanTask) -> tuple[SectionResult, list[BaseMessage]]:
        """执行一条计划任务，并返回：

        - SectionResult：该分区的最终展示内容 + 原始结构化结果
        - trace_messages：用于学习/调试的消息轨迹（包含 ToolMessage）

        注意：trace_messages 进入 LangGraph state 后，会被 `compress_messages()` 处理。
        """

        trace_messages: list[BaseMessage] = []
        last_result: dict[str, Any] = {}
        last_source = "mcp"

        for attempt in range(1, self._max_tool_attempts + 1):
            trace_messages.append(
                AIMessage(
                    content=(
                        f"[{task['display_name']}] 第 {attempt} 轮思考："
                        f"先检查缓存，再决定是否调用 {task['tool_name']}。"
                    )
                )
            )

            # ReAct 风格里的 "Action" 不是盲目调工具，而是先查缓存。
            # 这样能把“工具复用”变成执行路径的一部分，而不是事后优化。
            cached = self._cache.lookup(task["service"], task["tool_name"], task["arguments"])
            if cached is not None:
                raw_result = cached
                last_source = "cache"
            else:
                # 学习版内部直接调用 FastMCP 的 tool；真实项目可在 McpClientManager 内替换。
                raw_result = await self._mcp_client.call_tool(task["service"], task["arguments"])
                self._cache.store(task["service"], task["tool_name"], task["arguments"], raw_result)
                last_source = "mcp"

            last_result = raw_result
            trace_messages.append(
                ToolMessage(
                    content=json.dumps(
                        {"source": last_source, "result": raw_result},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    tool_call_id=f"{task['section']}-{attempt}",
                )
            )

            # 这里的“观察”不是看模型心情，而是用固定字段判断任务是否完成。
            # 这样每个子 Agent 都有可解释的结束条件。
            if self._is_complete(task["section"], raw_result):
                trace_messages.append(
                    AIMessage(content=f"[{task['display_name']}] 观察：关键字段完整，本轮结束。")
                )
                return (
                    {
                        "section": task["section"],
                        "content": self._llm.render_section(task["section"], raw_result),
                        "source": last_source,
                        "raw_result": raw_result,
                    },
                    trace_messages,
                )

            trace_messages.append(
                AIMessage(content=f"[{task['display_name']}] 观察：结果不完整，继续下一轮补查。")
            )

        # 走到这里说明两轮都没有拿到完整结构，仍然把最后一次结果交回父 Agent。
        # 这是学习版里一个重要取舍：允许局部任务降级，而不是让整份攻略直接失败。
        return (
            {
                "section": task["section"],
                "content": self._llm.render_section(task["section"], last_result),
                "source": last_source,
                "raw_result": last_result,
            },
            trace_messages,
        )

    def _is_complete(self, section: str, raw_result: dict[str, Any]) -> bool:
        """判断工具结果是否“足够完整”。

        这里用最透明的“必填字段集合”作为完成标准：
        - 便于理解 ReAct 循环何时停止
        - 便于你以后替换为真实工具/真实模型后继续沿用该判断点
        """

        required_fields = {
            "transport_train": ("options", "advice"),
            "transport_flight": ("options", "advice"),
            "weather": ("summary", "temperature", "packing_tip"),
            "hotel": ("recommended_area", "options", "advice"),
            "scenic": ("spots", "route_tip"),
            "food": ("foods", "search_tip"),
        }
        return all(raw_result.get(field) for field in required_fields[section])
