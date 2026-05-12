"""真实场景版 LLM 适配层。

本文件保留 `LearningLLM` 类名和公开方法签名，让现有 workflow / worker 不用改。
内部从“完全写死的规则逻辑”改为“真实模型优先 + 工程校验 + 兜底降级”。

配置环境变量：
- TRAVEL_ASSISTANT_LLM_API_KEY：模型服务 API Key。
- TRAVEL_ASSISTANT_LLM_BASE_URL：OpenAI-compatible base url，默认 https://api.openai.com/v1。
- TRAVEL_ASSISTANT_LLM_MODEL：模型名，默认 gpt-4o-mini。
- TRAVEL_ASSISTANT_LLM_TIMEOUT：请求超时时间，默认 30 秒。
- TRAVEL_ASSISTANT_LLM_DISABLE_FALLBACK：设为 true 后，模型失败会直接抛错。

设计重点：
- Plan / Feedback Scope / Render / Replan 都优先交给真实 LLM。
- service 和 tool_name 仍由代码白名单控制，避免模型发明不存在的工具。
- 模型输出必须经过 JSON 解析、section 白名单、参数重建等校验。
- 没有 API Key 或模型异常时默认降级，保证学习项目仍然能跑通。
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

import httpx
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from app.models.state import ParsedRequest, PlanTask, SectionResult

CANONICAL_SECTIONS = (
    "transport_train",
    "transport_flight",
    "weather",
    "hotel",
    "scenic",
    "food",
)

SECTION_META = {
    "transport_train": {
        "display_name": "交通-火车",
        "service": "train",
        "tool_name": "search_train_tickets",
        "goal": "查询火车/高铁方案",
    },
    "transport_flight": {
        "display_name": "交通-航班",
        "service": "flight",
        "tool_name": "search_flights",
        "goal": "查询航班方案",
    },
    "weather": {
        "display_name": "天气",
        "service": "weather",
        "tool_name": "get_weather_forecast",
        "goal": "查询天气与穿衣建议",
    },
    "hotel": {
        "display_name": "住宿",
        "service": "hotel",
        "tool_name": "search_hotels",
        "goal": "查询住宿方案",
    },
    "scenic": {
        "display_name": "景点",
        "service": "map",
        "tool_name": "search_scenic_spots",
        "goal": "查询景点与简单路线",
    },
    "food": {
        "display_name": "美食",
        "service": "search",
        "tool_name": "web_search",
        "goal": "查询当地美食建议",
    },
}

PUBLIC_SECTION_ORDER = ("交通", "住宿", "天气", "景点", "美食")


class LLMConfigurationError(RuntimeError):
    """模型配置错误，例如缺少 TRAVEL_ASSISTANT_LLM_API_KEY。"""


class LLMResponseError(RuntimeError):
    """模型返回内容不符合当前节点要求。"""


class LearningLLM:
    """真实场景版父 Agent / LLM Gateway。

    输入例子：
        parsed_request = {
            "origin": "北京",
            "destination": "上海",
            "departure_date": "2026-05-01",
            "return_date": "2026-05-04",
            "preferences": "预算有限，少走路",
        }

    输出例子：
        build_plan(parsed_request) -> [
            {
                "section": "transport_train",
                "display_name": "交通-火车",
                "service": "train",
                "tool_name": "search_train_tickets",
                "goal": "查询火车/高铁方案",
                "arguments": {...},
            },
            ...
        ]

    职责说明：
        父 Agent 不直接查车票、酒店或天气，只负责规划、反馈范围判断、
        工具结果改写和最终攻略整合。
    """

    def __init__(self) -> None:
        """初始化真实模型配置和 Prompt 模板。

        输入例子：
            TRAVEL_ASSISTANT_LLM_API_KEY 已配置。
            TRAVEL_ASSISTANT_LLM_MODEL = "gpt-4o-mini"。

        输出例子：
            self._model = "gpt-4o-mini"。
            self.plan_prompt = ChatPromptTemplate(...)。

        注意：
            初始化阶段不请求模型，避免服务启动时因网络或密钥问题直接失败。
        """

        self._api_key = os.getenv("TRAVEL_ASSISTANT_LLM_API_KEY", "").strip()
        self._base_url = os.getenv("TRAVEL_ASSISTANT_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self._model = os.getenv("TRAVEL_ASSISTANT_LLM_MODEL", "gpt-4o-mini").strip()
        self._timeout = float(os.getenv("TRAVEL_ASSISTANT_LLM_TIMEOUT", "30"))
        self._disable_fallback = os.getenv("TRAVEL_ASSISTANT_LLM_DISABLE_FALLBACK", "").lower() in {
            "1",
            "true",
            "yes",
        }

        self.plan_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是旅游规划父 Agent。必须输出 JSON，不要发明不存在的工具。"),
                (
                    "human",
                    "已解析旅行请求：\n{parsed_request}\n\n"
                    "工具目录：\n{tool_catalog}\n\n"
                    "用户反馈：{feedback_text}\n"
                    "本轮允许调整的分区：{allowed_sections}\n\n"
                    "输出 JSON：{{\"tasks\":[{{\"section\":\"hotel\",\"goal\":\"...\",\"arguments\":{{\"preference\":\"...\"}}}}]}}",
                ),
            ]
        )
        self.feedback_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是反馈影响范围分析器。必须输出 JSON。"),
                (
                    "human",
                    "用户反馈：{feedback}\n"
                    "可选分区：transport_train, transport_flight, weather, hotel, scenic, food。\n"
                    "输出 JSON：{{\"affected_sections\":[\"hotel\"]}}。",
                ),
            ]
        )
        self.section_render_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是旅游攻略编辑。只能基于工具 JSON 写建议，不要编造票价、班次、酒店、天气。",
                ),
                (
                    "human",
                    "分区：{section}\n工具结果 JSON：\n{raw_result}\n\n"
                    "请输出该分区正文，使用 Markdown 列表，简洁可执行。",
                ),
            ]
        )
        self.replan_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是最终攻略整合器。保留输入事实，不新增工具结果外的具体信息。",
                ),
                (
                    "human",
                    "旅行请求：\n{parsed_request}\n\n"
                    "已有分区：\n{sections}\n\n"
                    "请输出完整 Markdown 攻略。标题格式：# 目的地旅游攻略。"
                    "分区顺序：交通、住宿、天气、景点、美食。",
                ),
            ]
        )

    def build_plan(
        self,
        parsed_request: ParsedRequest,
        *,
        feedback_text: str = "",
        affected_sections: list[str] | None = None,
    ) -> list[PlanTask]:
        """Plan 节点：把旅行请求拆成专业工具任务。

        输入例子：
            parsed_request = {
                "origin": "北京",
                "destination": "上海",
                "departure_date": "2026-05-01",
                "return_date": "2026-05-04",
                "preferences": "住宿便宜一点",
            }
            feedback_text = ""
            affected_sections = None

        输出例子：
            [
                {"section": "transport_train", "service": "train", "tool_name": "search_train_tickets", "arguments": {...}},
                {"section": "hotel", "service": "hotel", "tool_name": "search_hotels", "arguments": {...}},
            ]

        真实逻辑：
            先让 LLM 输出任务 JSON，再由代码白名单校验 section 和工具，
            最后重建 arguments，避免模型把不存在的参数传给 MCP。
        """

        allowed_sections = self._normalise_sections(affected_sections) if affected_sections else list(CANONICAL_SECTIONS)
        try:
            prompt_messages = self.plan_prompt.format_messages(
                parsed_request=json.dumps(parsed_request, ensure_ascii=False, indent=2),
                tool_catalog=json.dumps(SECTION_META, ensure_ascii=False, indent=2),
                feedback_text=feedback_text or "无",
                allowed_sections=", ".join(allowed_sections),
            )
            payload = self._complete_json(prompt_messages)
            tasks = payload.get("tasks")
            if not isinstance(tasks, list):
                raise LLMResponseError("Plan 输出缺少 tasks 数组。")
            return self._validate_plan_tasks(tasks, parsed_request, feedback_text, allowed_sections)
        except Exception:
            if self._disable_fallback:
                raise
            return self._fallback_plan(parsed_request, feedback_text=feedback_text, allowed_sections=allowed_sections)

    def plan_messages(self, request_text: str, tasks: list[PlanTask]) -> list[BaseMessage]:
        """生成 Plan 阶段写入 LangGraph state 的消息。

        输入例子：
            request_text = "我想从北京去上海旅游"
            tasks = [{"display_name": "住宿", "goal": "查询住宿方案"}]

        输出例子：
            [SystemMessage(...), HumanMessage("原始需求：... 本轮任务：- 住宿：查询住宿方案")]
        """

        task_lines = "\n".join(f"- {task['display_name']}：{task['goal']}" for task in tasks)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是旅游规划父 Agent，以下是你刚刚生成的任务计划。"),
                ("human", "原始需求：{request_text}\n本轮任务：\n{task_lines}"),
            ]
        )
        return prompt.format_messages(request_text=request_text, task_lines=task_lines)

    def feedback_messages(self, feedback: str, affected_sections: list[str]) -> list[BaseMessage]:
        """生成 Feedback Scope 阶段写入 state 的消息。

        输入例子：
            feedback = "住宿想更便宜一点，其他不动"
            affected_sections = ["hotel"]

        输出例子：
            [SystemMessage(...), HumanMessage("用户反馈：住宿想更便宜一点... 识别结果：hotel")]
        """

        return self.feedback_prompt.format_messages(feedback=f"{feedback}\n识别结果：{affected_sections}")

    def replan_messages(self, destination: str, sections: Iterable[str]) -> list[BaseMessage]:
        """生成 Replan 阶段写入 state 的消息。

        输入例子：
            destination = "上海"
            sections = ["交通", "住宿", "天气"]

        输出例子：
            [SystemMessage(...), HumanMessage("目的地：上海，可用分区：交通、住宿、天气")]
        """

        section_labels = "、".join(sections)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你负责把多 Agent 结果整理成最终攻略。"),
                ("human", "目的地：{destination}\n可用分区：{sections}"),
            ]
        )
        return prompt.format_messages(destination=destination, sections=section_labels)

    def determine_feedback_scope(self, feedback: str) -> list[str]:
        """Feedback Scope 节点：判断用户反馈影响哪些分区。

        输入例子：
            feedback = "住宿想更便宜一点，景点不要太分散"

        输出例子：
            ["hotel", "scenic"]
        """

        try:
            payload = self._complete_json(self.feedback_prompt.format_messages(feedback=feedback))
            sections = payload.get("affected_sections")
            if not isinstance(sections, list):
                raise LLMResponseError("Feedback 输出缺少 affected_sections 数组。")
            normalised = self._normalise_sections(sections)
            return normalised or list(CANONICAL_SECTIONS)
        except Exception:
            if self._disable_fallback:
                raise
            return self._fallback_feedback_scope(feedback)

    def render_section(self, section: str, raw_result: dict[str, object]) -> str:
        """Render 节点：把单个工具结果改写成攻略分区正文。

        输入例子：
            section = "hotel"
            raw_result = {"recommended_area": "上海市中心", "options": [{"name": "上海精选酒店"}]}

        输出例子：
            "建议优先住在上海市中心：\n- 上海精选酒店：..."
        """

        try:
            text = self._chat_text(
                self.section_render_prompt.format_messages(
                    section=section,
                    raw_result=json.dumps(raw_result, ensure_ascii=False, indent=2),
                )
            ).strip()
            if not text:
                raise LLMResponseError("Render 输出为空。")
            return text
        except Exception:
            if self._disable_fallback:
                raise
            return self._fallback_render_section(section, raw_result)

    def compose_guide_sections(
        self,
        parsed_request: ParsedRequest,
        agent_outputs: dict[str, SectionResult],
        previous_sections: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Replan 子节点：把子 Agent 输出合并成固定分区。

        输入例子：
            agent_outputs = {"transport_train": {"content": "..."}, "hotel": {"content": "..."}}
            previous_sections = {"天气": "旧天气内容"}

        输出例子：
            {"交通": "### 火车\n...", "住宿": "...", "天气": "旧天气内容"}
        """

        _ = parsed_request
        sections = dict(previous_sections or {})
        transport_parts: list[str] = []
        if "transport_train" in agent_outputs:
            transport_parts.append("### 火车\n" + agent_outputs["transport_train"]["content"])
        if "transport_flight" in agent_outputs:
            transport_parts.append("### 航班\n" + agent_outputs["transport_flight"]["content"])
        if transport_parts:
            sections["交通"] = "\n\n".join(transport_parts)

        mapping = {"hotel": "住宿", "weather": "天气", "scenic": "景点", "food": "美食"}
        for internal_section, public_title in mapping.items():
            if internal_section in agent_outputs:
                sections[public_title] = agent_outputs[internal_section]["content"]
        return sections

    def compose_guide_markdown(self, parsed_request: ParsedRequest, sections: dict[str, str]) -> str:
        """Replan 节点：把固定分区合成最终 Markdown 攻略。

        输入例子：
            parsed_request = {"origin": "北京", "destination": "上海", "departure_date": "2026-05-01", "return_date": "2026-05-04"}
            sections = {"交通": "...", "住宿": "..."}

        输出例子：
            "# 上海旅游攻略\n\n出发地：北京\n出行时间：2026-05-01 至 2026-05-04\n\n## 交通\n..."
        """

        try:
            text = self._chat_text(
                self.replan_prompt.format_messages(
                    parsed_request=json.dumps(parsed_request, ensure_ascii=False, indent=2),
                    sections=json.dumps(sections, ensure_ascii=False, indent=2),
                )
            ).strip()
            if not text.startswith("#"):
                raise LLMResponseError("最终攻略不是 Markdown 标题开头。")
            return text
        except Exception:
            if self._disable_fallback:
                raise
            return self._fallback_compose_guide_markdown(parsed_request, sections)

    def _chat_text(self, messages: list[BaseMessage]) -> str:
        """底层模型调用：发送 Chat Completions 请求并返回文本。

        输入例子：
            messages = [SystemMessage("你是..."), HumanMessage("请输出 JSON")]

        输出例子：
            "{\"affected_sections\":[\"hotel\"]}"
        """

        if not self._api_key:
            raise LLMConfigurationError("缺少 TRAVEL_ASSISTANT_LLM_API_KEY，无法调用真实 LLM。")
        payload = {
            "model": self._model,
            "messages": [self._to_openai_message(message) for message in messages],
            "temperature": 0.2,
        }
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(f"模型响应结构异常：{data}") from exc

    def _complete_json(self, messages: list[BaseMessage]) -> dict[str, Any]:
        """底层 JSON 调用：要求模型文本可解析为 JSON 对象。

        输入例子：
            messages = [..., HumanMessage("输出 JSON")]

        输出例子：
            {"tasks": [{"section": "hotel"}]}
        """

        text = self._strip_json_fence(self._chat_text(messages))
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"模型没有返回合法 JSON：{text}") from exc
        if not isinstance(payload, dict):
            raise LLMResponseError("模型 JSON 顶层必须是 object。")
        return payload

    def _validate_plan_tasks(
        self,
        raw_tasks: list[Any],
        parsed_request: ParsedRequest,
        feedback_text: str,
        allowed_sections: list[str],
    ) -> list[PlanTask]:
        """校验模型生成的任务，防止坏 JSON 污染执行层。

        输入例子：
            raw_tasks = [{"section": "hotel", "goal": "找便宜住宿", "arguments": {"preference": "经济型"}}]

        输出例子：
            [{"section": "hotel", "display_name": "住宿", "service": "hotel", "tool_name": "search_hotels", "arguments": {...}}]
        """

        tasks: list[PlanTask] = []
        allowed = set(allowed_sections)
        for item in raw_tasks:
            if not isinstance(item, dict):
                continue
            section = item.get("section")
            if section not in CANONICAL_SECTIONS or section not in allowed:
                continue
            meta = SECTION_META[section]
            model_arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            tasks.append(
                {
                    "section": section,
                    "display_name": meta["display_name"],
                    "service": meta["service"],
                    "tool_name": meta["tool_name"],
                    "goal": str(item.get("goal") or meta["goal"]),
                    "arguments": self._build_tool_arguments(section, parsed_request, feedback_text, model_arguments),
                }
            )
        if not tasks:
            raise LLMResponseError("模型没有生成任何可执行任务。")
        return tasks

    def _build_tool_arguments(
        self,
        section: str,
        parsed_request: ParsedRequest,
        feedback_text: str,
        model_arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """为指定 section 构造工具参数。

        输入例子：
            section = "hotel"
            model_arguments = {"preference": "经济型酒店"}

        输出例子：
            {"destination": "上海", "departure_date": "2026-05-01", "return_date": "2026-05-04", "preference": "经济型酒店"}
        """

        model_arguments = model_arguments or {}
        preference = "；".join(
            part
            for part in (
                str(parsed_request.get("preferences", "")).strip(),
                str(model_arguments.get("preference", "")).strip(),
                feedback_text.strip(),
            )
            if part
        )
        if section in ("transport_train", "transport_flight"):
            return {
                "origin": parsed_request["origin"],
                "destination": parsed_request["destination"],
                "departure_date": parsed_request["departure_date"],
                "return_date": parsed_request["return_date"],
                "preference": preference,
            }
        if section == "weather":
            return {
                "destination": parsed_request["destination"],
                "departure_date": parsed_request["departure_date"],
                "return_date": parsed_request["return_date"],
            }
        if section == "hotel":
            return {
                "destination": parsed_request["destination"],
                "departure_date": parsed_request["departure_date"],
                "return_date": parsed_request["return_date"],
                "preference": preference,
            }
        if section in ("scenic", "food"):
            return {"destination": parsed_request["destination"], "preference": preference}
        raise ValueError(f"未知 section：{section}")

    def _normalise_sections(self, sections: Iterable[Any]) -> list[str]:
        """把模型返回的分区名清洗成系统内部 section。

        输入例子：
            sections = ["住宿", "hotel", "景点"]

        输出例子：
            ["hotel", "scenic"]
        """

        alias = {
            "交通": ["transport_train", "transport_flight"],
            "火车": ["transport_train"],
            "高铁": ["transport_train"],
            "航班": ["transport_flight"],
            "飞机": ["transport_flight"],
            "住宿": ["hotel"],
            "酒店": ["hotel"],
            "天气": ["weather"],
            "景点": ["scenic"],
            "路线": ["scenic"],
            "美食": ["food"],
            "餐饮": ["food"],
        }
        result: list[str] = []
        for raw_section in sections:
            section = str(raw_section).strip()
            if section in CANONICAL_SECTIONS:
                result.append(section)
            else:
                result.extend(alias.get(section, []))
        return list(dict.fromkeys(result))

    def _to_openai_message(self, message: BaseMessage) -> dict[str, str]:
        """把 LangChain message 转成 Chat Completions message。

        输入例子：
            HumanMessage(content="你好")

        输出例子：
            {"role": "user", "content": "你好"}
        """

        message_type = getattr(message, "type", "human")
        role = "system" if message_type == "system" else "assistant" if message_type == "ai" else "user"
        return {"role": role, "content": str(message.content)}

    def _strip_json_fence(self, text: str) -> str:
        """移除模型可能返回的 Markdown JSON 代码块。

        输入例子：
            ```json
            {"tasks": []}
            ```

        输出例子：
            {"tasks": []}
        """

        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.removeprefix("```json").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned.removesuffix("```").strip()
        return cleaned

    def _fallback_plan(self, parsed_request: ParsedRequest, *, feedback_text: str, allowed_sections: list[str]) -> list[PlanTask]:
        """Plan 兜底：模型不可用时生成稳定任务。"""

        return [
            {
                "section": section,
                "display_name": SECTION_META[section]["display_name"],
                "service": SECTION_META[section]["service"],
                "tool_name": SECTION_META[section]["tool_name"],
                "goal": SECTION_META[section]["goal"],
                "arguments": self._build_tool_arguments(section, parsed_request, feedback_text),
            }
            for section in allowed_sections
        ]

    def _fallback_feedback_scope(self, feedback: str) -> list[str]:
        """Feedback Scope 兜底：模型不可用时用关键词做保守判断。"""

        scope: list[str] = []
        if any(keyword in feedback for keyword in ("交通", "高铁", "火车")):
            scope.append("transport_train")
        if any(keyword in feedback for keyword in ("交通", "航班", "飞机")):
            scope.append("transport_flight")
        if any(keyword in feedback for keyword in ("住宿", "酒店", "民宿")):
            scope.append("hotel")
        if any(keyword in feedback for keyword in ("天气", "下雨", "穿衣", "气温")):
            scope.append("weather")
        if any(keyword in feedback for keyword in ("景点", "路线", "打卡", "地图", "少走路")):
            scope.append("scenic")
        if any(keyword in feedback for keyword in ("美食", "小吃", "餐厅", "吃", "不要辣")):
            scope.append("food")
        return list(dict.fromkeys(scope)) or list(CANONICAL_SECTIONS)

    def _fallback_render_section(self, section: str, raw_result: dict[str, object]) -> str:
        """Render 兜底：模型不可用时按结构化字段直接渲染。"""

        if section == "transport_train":
            options = raw_result.get("options", [])
            lines = ["优先比较以下火车/高铁方案："]
            for option in options if isinstance(options, list) else []:
                lines.append(f"- {option['name']}：{option['route']}，耗时{option['duration']}，参考价{option['price']}")
            lines.append(f"建议：{raw_result.get('advice', '优先选择更贴合你时间安排的班次。')}")
            return "\n".join(lines)
        if section == "transport_flight":
            options = raw_result.get("options", [])
            lines = ["可以同步比较以下航班方案："]
            for option in options if isinstance(options, list) else []:
                lines.append(f"- {option['name']}：{option['route']}，耗时{option['duration']}，参考价{option['price']}")
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
            options = raw_result.get("options", [])
            for option in options if isinstance(options, list) else []:
                lines.append(f"- {option['name']}：{option['price']}，{option['reason']}")
            lines.append(f"建议：{raw_result.get('advice', '优先考虑交通衔接顺畅的位置。')}")
            return "\n".join(lines)
        if section == "scenic":
            lines = [f"景点建议以 {raw_result.get('focus_area', '市中心')} 为主线："]
            spots = raw_result.get("spots", [])
            for spot in spots if isinstance(spots, list) else []:
                lines.append(f"- {spot}")
            lines.append(f"路线提示：{raw_result.get('route_tip', '把同区域景点放在同一天更省时间。')}")
            return "\n".join(lines)
        if section == "food":
            lines = ["美食建议可以优先看这些方向："]
            foods = raw_result.get("foods", [])
            for item in foods if isinstance(foods, list) else []:
                lines.append(f"- {item}")
            lines.append(f"搜索补充：{raw_result.get('search_tip', '优先就近安排，不必跨城折返。')}")
            return "\n".join(lines)
        return "当前分区暂无可展示内容。"

    def _fallback_compose_guide_markdown(self, parsed_request: ParsedRequest, sections: dict[str, str]) -> str:
        """Replan 兜底：模型不可用时稳定拼接 Markdown。"""

        title = f"# {parsed_request['destination']}旅游攻略"
        meta = f"出发地：{parsed_request['origin']}\n出行时间：{parsed_request['departure_date']} 至 {parsed_request['return_date']}"
        body_parts = [title, "", meta]
        for section_title in PUBLIC_SECTION_ORDER:
            content = sections.get(section_title)
            if content:
                body_parts.extend(["", f"## {section_title}", content])
        return "\n".join(body_parts).strip()
