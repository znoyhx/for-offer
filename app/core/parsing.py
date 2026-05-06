"""把自然语言旅游需求解析成结构化字段。

这个解析器故意保持简单透明：
- 不追求“什么都能识别”。
- 只覆盖这个学习项目真正需要的字段。
- 识别不到时，直接触发 LangGraph interrupt 让用户补充。
"""

from __future__ import annotations

import re
from datetime import datetime

from app.models.state import ParsedRequest

FIELD_LABELS = {
    "origin": "出发地",
    "destination": "目的地",
    "departure_date": "出发日期",
    "return_date": "返程日期",
}

ROUTE_ENDINGS = (
    "旅游",
    "旅行",
    "玩",
    "度假",
    "攻略",
    "行程",
    "出差",
)


class TravelRequestParser:
    """最小化请求解析器。

    它的目标不是“完美理解所有自然语言”，而是服务于工作流：
    - 尽可能从文本里抽出 4 个必需字段：出发地/目的地/出发日期/返程日期
    - 抽不出来就交给 `ensure_requirements` 触发 interrupt 让用户补
    - 同时从文本里提取少量偏好词，透传给工具参数
    """

    route_pattern = re.compile(
        r"从(?P<origin>[\u4e00-\u9fffA-Za-z]{2,20}?)(?:出发)?(?:去|到)(?P<destination>[\u4e00-\u9fffA-Za-z]{2,20})"
    )
    origin_patterns = (
        re.compile(r"出发地[:：]?(?P<value>[\u4e00-\u9fffA-Za-z]{2,20})"),
        re.compile(r"从(?P<value>[\u4e00-\u9fffA-Za-z]{2,20})(?:出发|去|到)"),
    )
    destination_patterns = (
        re.compile(r"目的地[:：]?(?P<value>[\u4e00-\u9fffA-Za-z]{2,20})"),
        re.compile(r"(?:去|到)(?P<value>[\u4e00-\u9fffA-Za-z]{2,20})"),
    )
    date_pattern = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}月\d{1,2}日")
    preference_keywords = (
        "预算",
        "便宜",
        "舒适",
        "高铁",
        "火车",
        "飞机",
        "亲子",
        "自由行",
        "美食",
        "少走路",
        "拍照",
        "安静",
        "不要辣",
        "酒店",
    )

    def parse(self, text: str, base: ParsedRequest | None = None) -> ParsedRequest:
        """从自然语言文本中抽取当前图需要的字段。"""

        parsed: ParsedRequest = dict(base or {})
        cleaned = self._normalise_text(text)

        if not cleaned:
            return parsed

        route_match = self.route_pattern.search(cleaned)
        if route_match:
            # “从 A 去 B”是最直接的路径表达，命中后可一次性拿到起终点。
            parsed["origin"] = self._clean_city(route_match.group("origin"))
            parsed["destination"] = self._clean_city(route_match.group("destination"))

        if "origin" not in parsed:
            origin = self._match_city(self.origin_patterns, cleaned)
            if origin:
                parsed["origin"] = origin

        if "destination" not in parsed:
            destination = self._match_city(self.destination_patterns, cleaned)
            if destination:
                parsed["destination"] = destination

        extracted_dates = self._extract_dates(cleaned)
        if extracted_dates["departure_date"]:
            parsed["departure_date"] = extracted_dates["departure_date"]
        if extracted_dates["return_date"]:
            parsed["return_date"] = extracted_dates["return_date"]

        preferences = [keyword for keyword in self.preference_keywords if keyword in cleaned]
        if preferences:
            # 去重后拼回字符串，便于后续子 Agent 直接带着偏好调工具。
            parsed["preferences"] = " ".join(dict.fromkeys(preferences))

        return parsed

    def find_missing_fields(self, parsed: ParsedRequest) -> list[str]:
        """按固定顺序找出缺失字段。"""

        ordered_fields = ("origin", "destination", "departure_date", "return_date")
        return [field for field in ordered_fields if not parsed.get(field)]

    def build_missing_question(self, missing_fields: list[str]) -> str:
        """把缺失字段翻译成用户能直接回答的补充问题。"""

        labels = "、".join(FIELD_LABELS[field] for field in missing_fields)
        return (
            f"为了继续生成攻略，请补充：{labels}。"
            "你可以直接回答类似“出发地北京，目的地上海，2026-05-01出发，2026-05-04返回”。"
        )

    def _normalise_text(self, text: str) -> str:
        return (
            text.replace("\n", " ")
            .replace("，", ",")
            .replace("。", ".")
            .replace("：", ":")
            .strip()
        )

    def _match_city(self, patterns: tuple[re.Pattern[str], ...], text: str) -> str | None:
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return self._clean_city(match.group("value"))
        return None

    def _clean_city(self, value: str) -> str:
        city = value.strip(" ,.;，。；")
        for suffix in ROUTE_ENDINGS:
            if city.endswith(suffix):
                city = city[: -len(suffix)]
        return city.strip(" ,.;，。；")

    def _extract_dates(self, text: str) -> dict[str, str | None]:
        """优先用上下文词判断日期角色，再退回“第一个出发、第二个返程”的保底策略。"""

        departure_date: str | None = None
        return_date: str | None = None
        all_dates: list[str] = []

        for match in self.date_pattern.finditer(text):
            raw_date = match.group(0)
            normalised_date = self._normalise_date(raw_date)
            all_dates.append(normalised_date)

            context_before = text[max(0, match.start() - 8) : match.start()]
            context_after = text[match.end() : match.end() + 8]
            context = f"{context_before}{context_after}"

            # 用“返/回程/出发/开始”等上下文词，尽量推断日期是去程还是返程。
            # 这样用户只写“2026-05-01到2026-05-04”时也能被兜底逻辑覆盖。

            if any(keyword in context for keyword in ("返", "回程", "返回", "返程", "结束")):
                return_date = normalised_date
            elif any(keyword in context for keyword in ("出发", "去程", "启程", "开始")):
                departure_date = normalised_date

        if len(all_dates) >= 2:
            departure_date = departure_date or all_dates[0]
            return_date = return_date or all_dates[1]
        elif len(all_dates) == 1:
            departure_date = departure_date or all_dates[0]

        return {
            "departure_date": departure_date,
            "return_date": return_date,
        }

    def _normalise_date(self, raw_date: str) -> str:
        if "-" in raw_date:
            year, month, day = raw_date.split("-")
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

        month_str, day_str = raw_date.replace("日", "").split("月")
        current_year = datetime.now().year
        return f"{current_year:04d}-{int(month_str):02d}-{int(day_str):02d}"

