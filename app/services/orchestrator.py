"""应用级编排器（API <-> LangGraph 连接层）。

你可以把 `TravelAssistantService` 理解成“粘合层”：

- 向上：提供给 FastAPI 的 3 个方法（create/resume/feedback）。
- 向下：持有并驱动 LangGraph 图（Plan -> Execute -> Replan）。

这个模块刻意把概念边界写清楚：

- **会话**：用 `SessionStore` 做最小内存存储（草稿不要求数据库）。
- **工作流**：用 `build_workflow()` 生成图，图节点内部再调用 parser/llm/worker。
- **工具层**：MCP Client + Mock MCP 服务，模拟“按领域服务组织工具”。
- **缓存**：工具调用前先 lookup，相似命中直接复用。

学习路径（来自 docs/学习项目文档.md 推荐顺序）：
main -> orchestrator -> workflow -> worker -> mcp/mock -> cache/context
"""

from __future__ import annotations

import uuid
from copy import deepcopy

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agents.worker import TravelWorkerAgent
from app.config import settings
from app.core.cache import ToolResultCache
from app.core.llm import LearningLLM
from app.core.parsing import TravelRequestParser
from app.graph.workflow import build_workflow
from app.mcp.client import McpClientManager
from app.mcp.mock_servers import create_mock_mcp_services
from app.models.api import TravelResponse
from app.models.state import TravelAssistantState
from app.services.session_store import SessionStore


class TravelAssistantService:
    """把 API 层和 LangGraph 层连接起来。

    这里集中创建并持有所有“学习版组件”，确保：

    - 一次请求内多轮 interrupt/resume 用同一张图、同一套缓存
    - 反馈重规划可以基于历史状态增量执行
    - API 层不关心 LangGraph/LangChain/MCP 的细节
    """

    def __init__(self) -> None:
        """构建学习版的各层组件。

        这里的依赖关系对应文档的主链路：

        用户请求 -> parser -> LangGraph -> llm(父 Agent 规划/整理)
        -> worker(子 Agent ReAct) -> mcp client -> mock mcp tools
        """

        self._settings = settings
        self._parser = TravelRequestParser()
        self._llm = LearningLLM()
        self._cache = ToolResultCache(similarity_threshold=settings.cache_similarity_threshold)
        self._mcp_client = McpClientManager(create_mock_mcp_services())
        self._worker = TravelWorkerAgent(
            cache=self._cache,
            mcp_client=self._mcp_client,
            llm=self._llm,
            max_tool_attempts=settings.max_tool_attempts,
        )
        self._sessions = SessionStore()
        self._graph = build_workflow(
            parser=self._parser,
            llm=self._llm,
            worker=self._worker,
            settings=settings,
            checkpointer=InMemorySaver(),
        )

    def mounted_mcp_apps(self):
        """返回可挂载到 FastAPI 的 MCP 子应用集合。"""

        return self._mcp_client.mounted_apps()

    def attach_host_app(self, app) -> None:
        """保留主应用与 MCP 子应用的“归属”关系。

        学习版内部直接调用 `FastMCP.call_tool()`，因此这里不做额外处理；
        但接口保留能让你以后替换成真实 MCP 客户端时更顺滑。
        """

        self._mcp_client.attach_host_app(app)

    async def create_plan(self, request_text: str) -> TravelResponse:
        """创建一次新的旅游规划。

        返回值可能是：
        - `needs_input`：parser 发现缺字段，LangGraph interrupt 暂停等待补充
        - `completed`：完整攻略已生成
        """

        session_id = uuid.uuid4().hex
        self._sessions.create(session_id, request_text)

        initial_state: TravelAssistantState = {
            "session_id": session_id,
            "mode": "initial",
            "request_text": request_text,
            "parsed_request": {},
            "messages": [HumanMessage(content=request_text)],
            "agent_outputs": {},
            "guide_sections": {},
            "status": "created",
        }
        response, state = await self._run_graph(session_id=session_id, thread_id=session_id, graph_input=initial_state)

        if response.status == "completed" and state is not None:
            # 只有在图完成时才持久化“最终状态”，避免把中断态当成最终结果。
            self._sessions.save_state(session_id, state)
        return response

    async def resume_plan(self, session_id: str, reply: str) -> TravelResponse:
        """把用户补充的信息注入到 interrupt 点，继续执行图。"""

        if not self._sessions.exists(session_id):
            raise KeyError(f"Unknown session id: {session_id}")

        response, state = await self._run_graph(
            session_id=session_id,
            thread_id=session_id,
            graph_input=Command(resume=reply),
        )
        if response.status == "completed" and state is not None:
            self._sessions.save_state(session_id, state)
        return response

    async def apply_feedback(self, session_id: str, feedback: str) -> TravelResponse:
        """对已有攻略做反馈重规划。

        关键点：
        - 读取并复制上一轮完成状态（保证未受影响分区可复用）
        - `mode=feedback` 让图从 plan_tasks 开始（跳过 ensure_requirements）
        - 使用新的 thread_id，把“某次反馈修改”当作一条独立轨迹
        """

        if not self._sessions.exists(session_id):
            raise KeyError(f"Unknown session id: {session_id}")

        base_state = self._sessions.load_state(session_id)
        if base_state is None or not base_state.get("final_guide"):
            raise ValueError("当前 session 还没有完整攻略，不能做反馈重规划。")

        revision = self._sessions.next_revision(session_id)
        feedback_state = deepcopy(base_state)
        feedback_state.update(
            {
                "mode": "feedback",
                "feedback_text": feedback,
                "messages": list(base_state.get("messages", [])) + [HumanMessage(content=f"用户反馈：{feedback}")],
                "status": "feedback",
            }
        )

        # 反馈重规划使用新的 thread_id，目的是把“原始生成”和“某次反馈修改”
        # 视为两条独立的图执行轨迹，便于理解 LangGraph 的状态恢复边界。
        response, state = await self._run_graph(
            session_id=session_id,
            thread_id=f"{session_id}-feedback-{revision}",
            graph_input=feedback_state,
        )
        if response.status == "completed" and state is not None:
            self._sessions.save_state(session_id, state)
        return response

    async def _run_graph(
        self,
        *,
        session_id: str,
        thread_id: str,
        graph_input,
    ) -> tuple[TravelResponse, TravelAssistantState | None]:
        """统一驱动 LangGraph 执行，并把结果翻译成 API 的响应模型。

        这里做两件“翻译工作”：
        1) 收集 astream 的 chunk，用于检测 interrupt
        2) 从图里抓最终 state，用于输出 `final_guide` 或持久化
        """

        # thread_id 是 LangGraph 用来区分执行轨迹/检查点的关键标识。
        # 学习版里用 session_id 作为初始 thread，反馈时会派生新 thread。
        config = {"configurable": {"thread_id": thread_id}}
        chunks: list[dict] = []

        async for chunk in self._graph.astream(graph_input, config=config):
            chunks.append(chunk)

        # LangGraph 的 interrupt 会以 chunk 形式冒出来，这里把它翻译成 API 层
        # 更容易消费的 `needs_input` 响应。
        interrupt_payload = self._extract_interrupt_payload(chunks)
        snapshot = await self._graph.aget_state(config)
        state = dict(snapshot.values) if snapshot and snapshot.values else None

        if interrupt_payload is not None:
            return (
                TravelResponse(
                    session_id=session_id,
                    status="needs_input",
                    question=interrupt_payload["question"],
                ),
                state,
            )

        guide = state.get("final_guide") if state else None
        return (
            TravelResponse(
                session_id=session_id,
                status="completed",
                guide=guide,
            ),
            state,
        )

    def _extract_interrupt_payload(self, chunks: list[dict]) -> dict[str, object] | None:
        """从 LangGraph chunk 中提取 interrupt 数据。

        LangGraph 的 interrupt 事件是一个“携带 value 的对象”；
        学习版统一把 value 转成 `{"question": ...}` 的结构，方便 API 层透出。
        """

        for chunk in chunks:
            interrupts = chunk.get("__interrupt__")
            if interrupts:
                interrupt_event = interrupts[0]
                value = interrupt_event.value
                if isinstance(value, dict):
                    return value
                return {"question": str(value)}
        return None
