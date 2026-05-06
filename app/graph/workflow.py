"""Plan -> Execute -> Replan 图定义。

这张图就是学习项目的“主流程地图”，对应 docs/学习项目文档.md 里的链路：

1. ensure_requirements：字段不够就 interrupt，等用户补充
2. plan_tasks：父 Agent 把请求拆成多个专业任务（Plan）
3. execute_tasks：子 Agent 逐个执行任务（Execute / ReAct 风格）
4. compose_guide：把结果整理成结构化攻略（Replan）

图节点都遵循同一原则：
- 输入是 `TravelAssistantState`
- 输出是“增量更新的 state patch”
- 每个节点边界都会做一次 `compress_messages()`，控制上下文成本
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import interrupt

from app.agents.worker import TravelWorkerAgent
from app.config import Settings
from app.core.context import compress_messages
from app.core.llm import LearningLLM
from app.core.parsing import TravelRequestParser
from app.models.state import TravelAssistantState


def build_workflow(
    *,
    parser: TravelRequestParser,
    llm: LearningLLM,
    worker: TravelWorkerAgent,
    settings: Settings,
    checkpointer,
):
    """创建并编译 LangGraph 图。

    参数都是“可替换的组件”，便于学习时做局部实验：

    - parser：需求解析与缺失字段判断
    - llm：父 Agent 规划/重规划/汇总的教学版实现
    - worker：子 Agent 执行器（缓存 + MCP 调用 + 完整性判断）
    - settings：控制上下文、缓存相似度、工具尝试次数等
    - checkpointer：LangGraph 检查点（本项目用 InMemorySaver）
    """

    async def ensure_requirements(state: TravelAssistantState) -> TravelAssistantState:
        """缺失信息闸门：不够就 interrupt。

        这个节点体现学习项目的一个关键取舍：
        与其让系统“胡乱猜”，不如明确暂停并向用户提问。
        """

        parsed = parser.parse(state.get("request_text", ""), state.get("parsed_request"))
        messages = list(state.get("messages", []))
        pending_question = ""

        while True:
            # 这里故意做成 "检查 -> interrupt -> 用户补充 -> 再检查" 的循环，
            # 对应 draft.txt 里的“信息不足时暂停请求用户补充”。
            missing_fields = parser.find_missing_fields(parsed)
            if not missing_fields:
                break

            pending_question = parser.build_missing_question(missing_fields)
            reply = interrupt(
                {
                    "question": pending_question,
                    "missing_fields": missing_fields,
                }
            )
            messages.extend(
                [
                    AIMessage(content=pending_question),
                    HumanMessage(content=str(reply)),
                ]
            )
            parsed = parser.parse(str(reply), parsed)

        return {
            "parsed_request": parsed,
            "missing_fields": [],
            "pending_question": pending_question,
            "messages": compress_messages(
                messages,
                recent_non_tool=settings.recent_message_limit,
                max_tokens=settings.max_context_tokens,
            ),
            "status": "ready",
        }

    async def plan_tasks(state: TravelAssistantState) -> TravelAssistantState:
        """Plan 阶段：父 Agent 生成任务列表。

        - 初始模式：生成全量 6 个任务
        - 反馈模式：先确定受影响分区，再只生成这些分区的新任务（增量重规划入口）
        """

        messages = list(state.get("messages", []))
        mode = state.get("mode", "initial")

        if mode == "feedback":
            # 反馈模式不会把整张图从头重做，而是先算出“受影响分区”，
            # 再只为这些分区生成新任务。这是增量重规划的入口。
            feedback = state["feedback_text"]
            affected_sections = llm.determine_feedback_scope(feedback)
            messages.append(HumanMessage(content=f"用户反馈：{feedback}"))
            messages.extend(llm.feedback_messages(feedback, affected_sections))
            plan = llm.build_plan(
                state["parsed_request"],
                feedback_text=feedback,
                affected_sections=affected_sections,
            )
        else:
            affected_sections = []
            plan = llm.build_plan(state["parsed_request"])
            messages.extend(llm.plan_messages(state["request_text"], plan))

        return {
            "plan": plan,
            "affected_sections": affected_sections,
            "messages": compress_messages(
                messages,
                recent_non_tool=settings.recent_message_limit,
                max_tokens=settings.max_context_tokens,
            ),
            "status": "planned",
        }

    async def execute_tasks(state: TravelAssistantState) -> TravelAssistantState:
        """Execute 阶段：把每个任务交给子 Agent 执行。

        这里不会“在一个节点里完成所有事”，而是：
        - 逐任务执行
        - 每个任务产出工具追踪消息（ToolMessage）
        - 每轮执行后压缩上下文
        """

        messages = list(state.get("messages", []))
        outputs = dict(state.get("agent_outputs", {}))

        for task in state.get("plan", []):
            # 父 Agent 在上一步只负责“拆任务”，这里才真正进入专业子 Agent 执行阶段。
            result, trace_messages = await worker.run(task)
            outputs[result["section"]] = result
            messages.extend(trace_messages)
            messages = compress_messages(
                messages,
                recent_non_tool=settings.recent_message_limit,
                max_tokens=settings.max_context_tokens,
            )

        return {
            "agent_outputs": outputs,
            "messages": messages,
            "status": "executed",
        }

    def compose_guide(state: TravelAssistantState) -> TravelAssistantState:
        """Replan 阶段：整理与合并子 Agent 输出，形成最终攻略。"""

        guide_sections = llm.compose_guide_sections(
            state["parsed_request"],
            state.get("agent_outputs", {}),
            previous_sections=state.get("guide_sections"),
        )
        final_guide = llm.compose_guide_markdown(state["parsed_request"], guide_sections)

        messages = list(state.get("messages", []))
        messages.extend(llm.replan_messages(state["parsed_request"]["destination"], guide_sections.keys()))

        return {
            "guide_sections": guide_sections,
            "final_guide": final_guide,
            "messages": compress_messages(
                messages,
                recent_non_tool=settings.recent_message_limit,
                max_tokens=settings.max_context_tokens,
            ),
            "status": "completed",
        }

    graph = StateGraph(TravelAssistantState)
    graph.add_node("ensure_requirements", ensure_requirements)
    graph.add_node("plan_tasks", plan_tasks)
    graph.add_node("execute_tasks", execute_tasks)
    graph.add_node("compose_guide", compose_guide)

    graph.add_conditional_edges(
        START,
        lambda state: "feedback" if state.get("mode") == "feedback" else "initial",
        {
            # 初始请求要先补齐信息；反馈请求已经有完整基础状态，
            # 所以可以直接进入计划阶段。
            "initial": "ensure_requirements",
            "feedback": "plan_tasks",
        },
    )
    graph.add_edge("ensure_requirements", "plan_tasks")
    graph.add_edge("plan_tasks", "execute_tasks")
    graph.add_edge("execute_tasks", "compose_guide")
    graph.add_edge("compose_guide", END)

    return graph.compile(checkpointer=checkpointer)
