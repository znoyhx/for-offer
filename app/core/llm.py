"""教学版 LLM 适配层。

为了保证项目“开箱可跑”，这里不强依赖真实大模型服务，
而是把“父 Agent 的拆解能力”和“Replan 的整理能力”收敛成
一组可读、可替换的规则方法。

重点不是伪装成真 LLM，而是把：
- LangChain 的 Prompt 组织方式
- LangGraph 的状态流转
- MCP 工具调用结果的整理方式

都保留出来，方便后续替换成真实模型。
"""

from __future__ import annotations

from typing import Iterable

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from app.models.state import ParsedRequest, PlanTask, SectionResult


class LearningLLM:
    """用规则实现的最小“LLM 接口”（父 Agent）。

    为什么要这样写？

        - 学习项目需要“开箱可跑”，不依赖外部模型服务与密钥。
        - 但又要保留 LLM 系统里真正重要的结构：
            - Plan：把需求拆成任务
            - Replan：把工具结果整理成结构化输出
            - Feedback：识别影响范围并做增量重规划

    因此本类用稳定可读的规则，模拟父 Agent 的三个核心职责。
    你后续替换成真实模型时，尽量保持这些方法签名不变即可。
    """

    def __init__(self) -> None:
        """构建教学版 prompt 模板。

        这里仍然使用 LangChain 的 `ChatPromptTemplate`：
        - 让你能学到 prompt 组织方式
        - 也为未来替换真实 ChatModel 留出位置
        """

        self.plan_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个教学用旅游规划父 Agent，只负责把任务拆给专业子 Agent。"),
                ("human", "原始需求：{request_text}\n本轮任务：\n{task_lines}"),
            ]
        )
        self.feedback_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你只做局部改写，不重做未被点名的部分。"),
                ("human", "用户反馈：{feedback}\n需要调整的分区：{sections}"),
            ]
        )
        self.replan_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你负责把多 Agent 结果整理成一份简洁攻略。"),
                ("human", "目的地：{destination}\n可用分区：{sections}"),
            ]
        )

    def build_plan(
        self,
        parsed_request: ParsedRequest,
        *,
        feedback_text: str = "",
        affected_sections: list[str] | None = None,
    ) -> list[PlanTask]:
        """把请求拆成 6 类工具任务。

        这就是“父 Agent + 6 个专业子 Agent”的落地点：
        - 每个任务绑定一个 MCP 服务（service）与一个工具名（tool_name）
        - arguments 统一准备好，子 Agent 执行器只负责“执行与检查”

        当 `affected_sections` 非空时，表示反馈模式，只返回受影响分区的任务。
        """

        preferences = feedback_text or parsed_request.get("preferences", "")
        # 偏好会直接透传到工具参数里，模拟“用户约束影响工具调用策略”。
        tasks = [
            self._task(
                section="transport_train",
                display_name="交通-火车",
                service="train",
                tool_name="search_train_tickets",
                goal="查询火车/高铁方案",
                arguments={
                    "origin": parsed_request["origin"],
                    "destination": parsed_request["destination"],
                    "departure_date": parsed_request["departure_date"],
                    "return_date": parsed_request["return_date"],
                    "preference": preferences,
                },
            ),
            self._task(
                section="transport_flight",
                display_name="交通-航班",
                service="flight",
                tool_name="search_flights",
                goal="查询航班方案",
                arguments={
                    "origin": parsed_request["origin"],
                    "destination": parsed_request["destination"],
                    "departure_date": parsed_request["departure_date"],
                    "return_date": parsed_request["return_date"],
                    "preference": preferences,
                },
            ),
            self._task(
                section="weather",
                display_name="天气",
                service="weather",
                tool_name="get_weather_forecast",
                goal="查询天气与穿衣建议",
                arguments={
                    "destination": parsed_request["destination"],
                    "departure_date": parsed_request["departure_date"],
                    "return_date": parsed_request["return_date"],
                },
            ),
            self._task(
                section="hotel",
                display_name="住宿",
                service="hotel",
                tool_name="search_hotels",
                goal="查询住宿方案",
                arguments={
                    "destination": parsed_request["destination"],
                    "departure_date": parsed_request["departure_date"],
                    "return_date": parsed_request["return_date"],
                    "preference": preferences,
                },
            ),
            self._task(
                section="scenic",
                display_name="景点",
                service="map",
                tool_name="search_scenic_spots",
                goal="查询景点与简单路线",
                arguments={
                    "destination": parsed_request["destination"],
                    "preference": preferences,
                },
            ),
            self._task(
                section="food",
                display_name="美食",
                service="search",
                tool_name="web_search",
                goal="查询当地美食建议",
                arguments={
                    "destination": parsed_request["destination"],
                    "preference": preferences,
                },
            ),
        ]

        if affected_sections:
            # 增量重规划：只生成受影响分区的任务，其它分区由 compose_guide_sections 复用旧内容。
            allowed = set(affected_sections)
            return [task for task in tasks if task["section"] in allowed]

        return tasks

    def plan_messages(self, request_text: str, tasks: list[PlanTask]) -> list[BaseMessage]:
        task_lines = "\n".join(f"- {task['display_name']}：{task['goal']}" for task in tasks)
        return self.plan_prompt.format_messages(request_text=request_text, task_lines=task_lines)

    def feedback_messages(self, feedback: str, affected_sections: list[str]) -> list[BaseMessage]:
        section_labels = "、".join(affected_sections)
        return self.feedback_prompt.format_messages(feedback=feedback, sections=section_labels)

    def replan_messages(self, destination: str, sections: Iterable[str]) -> list[BaseMessage]:
        section_labels = "、".join(sections)
        return self.replan_prompt.format_messages(destination=destination, sections=section_labels)

    def determine_feedback_scope(self, feedback: str) -> list[str]:
        """根据反馈关键词判断哪些部分需要重算。"""

        scope: list[str] = []

        if any(keyword in feedback for keyword in ("交通", "高铁", "火车")):
            scope.append("transport_train")
        if any(keyword in feedback for keyword in ("交通", "航班", "飞机")):
            scope.append("transport_flight")
        if any(keyword in feedback for keyword in ("住宿", "酒店", "民宿")):
            scope.append("hotel")
        if any(keyword in feedback for keyword in ("天气", "下雨", "穿衣", "气温")):
            scope.append("weather")
        if any(keyword in feedback for keyword in ("景点", "路线", "打卡", "地图")):
            scope.append("scenic")
        if any(keyword in feedback for keyword in ("美食", "小吃", "餐厅", "吃")):
            scope.append("food")

        # 如果没识别出关键词，就保守地重算全部分区，避免“反馈没生效”。
        if not scope:
            return [
                "transport_train",
                "transport_flight",
                "weather",
                "hotel",
                "scenic",
                "food",
            ]

        return list(dict.fromkeys(scope))

    def render_section(self, section: str, raw_result: dict[str, object]) -> str:
        """把结构化工具结果渲染成可阅读的分区文本。

        学习版刻意把“结果渲染”与“工具调用”分离：
        - 工具负责结构化数据
        - LLM/父 Agent 负责整理成最终呈现
        """

        if section == "transport_train":
            options = raw_result.get("options", [])
            lines = ["优先比较以下火车/高铁方案："]
            for option in options:
                lines.append(
                    f"- {option['name']}：{option['route']}，耗时{option['duration']}，参考价{option['price']}"
                )
            lines.append(f"建议：{raw_result.get('advice', '优先选择更贴合你时间安排的班次。')}")
            return "\n".join(lines)

        if section == "transport_flight":
            options = raw_result.get("options", [])
            lines = ["可以同步比较以下航班方案："]
            for option in options:
                lines.append(
                    f"- {option['name']}：{option['route']}，耗时{option['duration']}，参考价{option['price']}"
                )
            lines.append(f"建议：{raw_result.get('advice', '如果更重视效率，可优先看航班。')}")
            return "\n".join(lines)

        if section == "weather":
            return (
                f"{raw_result.get('date_range', '')} 预计 {raw_result.get('summary', '')}，"
                f"温度约 {raw_result.get('temperature', '')}。\n"
                f"穿衣建议：{raw_result.get('packing_tip', '')}"
            )

        if section == "hotel":
            lines = [f"建议优先住在 {raw_result.get('recommended_area', '核心城区')}："]
            for option in raw_result.get("options", []):
                lines.append(f"- {option['name']}：{option['price']}，{option['reason']}")
            lines.append(f"建议：{raw_result.get('advice', '优先考虑交通衔接顺畅的位置。')}")
            return "\n".join(lines)

        if section == "scenic":
            lines = [f"景点建议以 {raw_result.get('focus_area', '市中心')} 为主线："]
            for spot in raw_result.get("spots", []):
                lines.append(f"- {spot}")
            lines.append(f"路线提示：{raw_result.get('route_tip', '把同区域景点放在同一天更省时间。')}")
            return "\n".join(lines)

        if section == "food":
            lines = ["美食建议可以优先看这些方向："]
            for item in raw_result.get("foods", []):
                lines.append(f"- {item}")
            lines.append(f"搜索补充：{raw_result.get('search_tip', '优先就近安排，不必跨城折返。')}")
            return "\n".join(lines)

        return "当前分区暂无可展示内容。"

    def compose_guide_sections(
        self,
        parsed_request: ParsedRequest,
        agent_outputs: dict[str, SectionResult],
        previous_sections: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """把子 Agent 结果整理成最终展示分区。

        注意这里会把两个“交通子任务”合并成一个 `交通` 分区，
        其余分区一一映射，正好贴合草稿里的功能描述。
        """

        # `previous_sections` 用于反馈模式：未受影响分区直接复用，避免全量重算。
        sections = dict(previous_sections or {})
        transport_parts: list[str] = []

        if "transport_train" in agent_outputs:
            transport_parts.append("### 火车\n" + agent_outputs["transport_train"]["content"])
        if "transport_flight" in agent_outputs:
            transport_parts.append("### 航班\n" + agent_outputs["transport_flight"]["content"])
        if transport_parts:
            sections["交通"] = "\n\n".join(transport_parts)

        if "hotel" in agent_outputs:
            sections["住宿"] = agent_outputs["hotel"]["content"]
        if "weather" in agent_outputs:
            sections["天气"] = agent_outputs["weather"]["content"]
        if "scenic" in agent_outputs:
            sections["景点"] = agent_outputs["scenic"]["content"]
        if "food" in agent_outputs:
            sections["美食"] = agent_outputs["food"]["content"]

        return sections

    def compose_guide_markdown(self, parsed_request: ParsedRequest, sections: dict[str, str]) -> str:
        """把分区内容组装成最终 Markdown 文本。

        输出分区顺序固定为：交通/住宿/天气/景点/美食。
        这是学习项目的“稳定输出协议”，便于你观察局部重规划时哪些分区变化。
        """

        title = f"# {parsed_request['destination']}旅游攻略"
        meta = (
            f"出发地：{parsed_request['origin']}\n"
            f"出行时间：{parsed_request['departure_date']} 至 {parsed_request['return_date']}"
        )

        ordered_titles = ("交通", "住宿", "天气", "景点", "美食")
        body_parts = [title, "", meta]

        for section_title in ordered_titles:
            content = sections.get(section_title)
            if not content:
                continue
            body_parts.extend(["", f"## {section_title}", content])

        return "\n".join(body_parts).strip()

    def _task(
        self,
        *,
        section: str,
        display_name: str,
        service: str,
        tool_name: str,
        goal: str,
        arguments: dict[str, object],
    ) -> PlanTask:
        return {
            "section": section,
            "display_name": display_name,
            "service": service,
            "tool_name": tool_name,
            "goal": goal,
            "arguments": arguments,
        }

