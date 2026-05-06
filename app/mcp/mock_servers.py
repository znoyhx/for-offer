"""本地 Mock MCP 服务。

这个模块的目标不是“模拟真实平台”，而是提供稳定、可重复的结构化结果，
让你专注观察工作流本身：

父 Agent -> 子 Agent -> MCP Client -> MCP Tool -> 结构化结果 -> Replan 汇总

设计约束：
- 每个领域服务只暴露一个工具（边界清晰）
- 返回值结构稳定（便于 `_is_complete` 判断与渲染）
- 部分数字用 `_stable_number` 生成“看起来像真实数据但可重复”的结果
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp import FastMCP

SCENIC_SPOTS = {
    "上海": ["外滩", "武康路", "豫园", "徐汇滨江"],
    "北京": ["故宫", "景山", "什刹海", "前门"],
    "杭州": ["西湖", "灵隐寺", "河坊街", "龙井村"],
    "成都": ["宽窄巷子", "人民公园", "锦里", "东郊记忆"],
}

LOCAL_FOODS = {
    "上海": ["生煎", "小笼包", "葱油拌面"],
    "北京": ["烤鸭", "炸酱面", "铜锅涮肉"],
    "杭州": ["片儿川", "东坡肉", "龙井虾仁"],
    "成都": ["担担面", "钟水饺", "串串香"],
}


@dataclass(slots=True)
class MountedMcpService:
    """保存一个服务对外需要的最小信息。"""

    name: str
    tool_name: str
    server: FastMCP
    app: Any


def create_mock_mcp_services() -> dict[str, MountedMcpService]:
    """创建 6 个专业 MCP 服务。

    服务名与领域对应关系：
    - map：景点与路线
    - train：火车/高铁
    - flight：航班
    - weather：天气
    - hotel：住宿
    - search：美食/搜索
    """

    services: dict[str, MountedMcpService] = {}

    map_server = FastMCP(name="mock-map-service")

    @map_server.tool()
    def search_scenic_spots(destination: str, preference: str = "") -> dict[str, Any]:
        """返回目的地的景点清单与路线提示。"""

        focus_area = f"{destination}市中心"
        if "少走路" in preference:
            route_tip = "优先把核心景点压缩到同一区域，避免跨区来回折返。"
        elif "拍照" in preference:
            route_tip = "建议把更适合拍照的地标留在白天，步行街放在傍晚。"
        else:
            route_tip = "建议把同一区域景点放在同一天，减少交通切换。"

        spots = SCENIC_SPOTS.get(
            destination,
            [f"{destination}博物馆", f"{destination}老城区", f"{destination}城市公园", f"{destination}步行街"],
        )
        return {
            "destination": destination,
            "focus_area": focus_area,
            "spots": spots,
            "route_tip": route_tip,
        }

    services["map"] = MountedMcpService(
        name="map",
        tool_name="search_scenic_spots",
        server=map_server,
        app=map_server.streamable_http_app(),
    )

    train_server = FastMCP(name="mock-train-service")

    @train_server.tool()
    def search_train_tickets(
        origin: str,
        destination: str,
        departure_date: str,
        return_date: str,
        preference: str = "",
    ) -> dict[str, Any]:
        """返回火车/高铁方案（结构化 options + advice）。"""

        base = _stable_number(origin + destination + departure_date, 101, 199)
        advice = "如果你重视稳定与准点，可以优先考虑高铁。"
        if "高铁" in preference or "火车" in preference:
            advice = "你的反馈偏向轨道交通，建议优先锁定高铁方案。"
        return {
            "date_range": f"{departure_date} 至 {return_date}",
            "options": [
                {
                    "name": f"G{base}",
                    "route": f"{origin}南 -> {destination}虹桥",
                    "duration": f"{_stable_number(origin + destination, 4, 6)}小时{_stable_number(destination, 10, 50)}分",
                    "price": f"{_stable_number(origin + departure_date, 320, 680)}元",
                },
                {
                    "name": f"G{base + 8}",
                    "route": f"{origin}站 -> {destination}站",
                    "duration": f"{_stable_number(destination + origin, 5, 7)}小时{_stable_number(origin, 5, 45)}分",
                    "price": f"{_stable_number(destination + return_date, 300, 620)}元",
                },
            ],
            "advice": advice,
        }

    services["train"] = MountedMcpService(
        name="train",
        tool_name="search_train_tickets",
        server=train_server,
        app=train_server.streamable_http_app(),
    )

    flight_server = FastMCP(name="mock-flight-service")

    @flight_server.tool()
    def search_flights(
        origin: str,
        destination: str,
        departure_date: str,
        return_date: str,
        preference: str = "",
    ) -> dict[str, Any]:
        """返回航班方案（结构化 options + advice）。"""

        base = _stable_number(origin + destination + return_date, 600, 899)
        advice = "如果你更在意总耗时，航班通常更省时间。"
        if "便宜" in preference:
            advice = "你强调预算，可以把廉价时段航班一起纳入比较。"
        return {
            "date_range": f"{departure_date} 至 {return_date}",
            "options": [
                {
                    "name": f"MU{base}",
                    "route": f"{origin}机场 -> {destination}机场",
                    "duration": f"{_stable_number(origin + destination, 2, 3)}小时{_stable_number(return_date, 5, 45)}分",
                    "price": f"{_stable_number(departure_date + destination, 480, 1080)}元",
                },
                {
                    "name": f"CZ{base + 12}",
                    "route": f"{origin}机场 -> {destination}机场",
                    "duration": f"{_stable_number(destination + origin, 2, 4)}小时{_stable_number(origin, 0, 35)}分",
                    "price": f"{_stable_number(return_date + origin, 520, 1160)}元",
                },
            ],
            "advice": advice,
        }

    services["flight"] = MountedMcpService(
        name="flight",
        tool_name="search_flights",
        server=flight_server,
        app=flight_server.streamable_http_app(),
    )

    weather_server = FastMCP(name="mock-weather-service")

    @weather_server.tool()
    def get_weather_forecast(destination: str, departure_date: str, return_date: str) -> dict[str, Any]:
        """返回天气摘要、温度范围与穿衣建议。"""

        month = int(departure_date.split("-")[1])
        season = _season_summary(month)
        return {
            "destination": destination,
            "date_range": f"{departure_date} 至 {return_date}",
            "summary": season["summary"],
            "temperature": season["temperature"],
            "packing_tip": season["packing_tip"],
        }

    services["weather"] = MountedMcpService(
        name="weather",
        tool_name="get_weather_forecast",
        server=weather_server,
        app=weather_server.streamable_http_app(),
    )

    hotel_server = FastMCP(name="mock-hotel-service")

    @hotel_server.tool()
    def search_hotels(
        destination: str,
        departure_date: str,
        return_date: str,
        preference: str = "",
    ) -> dict[str, Any]:
        """返回住宿区域建议与候选酒店列表。"""

        if "便宜" in preference or "预算" in preference:
            recommended_area = f"{destination}交通便利商圈"
            options = [
                {"name": f"{destination}轻旅酒店", "price": "预算友好", "reason": "通勤成本低，适合学习版项目里的基础行程。"},
                {"name": f"{destination}快捷酒店", "price": "预算友好", "reason": "位置稳定，适合把预算留给交通和吃饭。"},
            ]
            advice = "你提到想更便宜一些，可以优先选择交通节点附近的经济型酒店。"
        elif "安静" in preference:
            recommended_area = f"{destination}次中心安静街区"
            options = [
                {"name": f"{destination}静居酒店", "price": "中等价位", "reason": "远离主干道，晚上更安静。"},
                {"name": f"{destination}花园酒店", "price": "中等价位", "reason": "步行环境更舒缓，适合慢节奏出行。"},
            ]
            advice = "如果你更重视休息质量，优先考虑相对安静的次中心街区。"
        else:
            recommended_area = f"{destination}市中心"
            options = [
                {"name": f"{destination}精选酒店", "price": "中等偏上", "reason": "景点与交通衔接更均衡。"},
                {"name": f"{destination}商务酒店", "price": "中等价位", "reason": "去景点和返程都比较顺手。"},
            ]
            advice = "默认建议住在市中心或交通换乘方便的区域。"

        return {
            "destination": destination,
            "date_range": f"{departure_date} 至 {return_date}",
            "recommended_area": recommended_area,
            "options": options,
            "advice": advice,
        }

    services["hotel"] = MountedMcpService(
        name="hotel",
        tool_name="search_hotels",
        server=hotel_server,
        app=hotel_server.streamable_http_app(),
    )

    search_server = FastMCP(name="mock-search-service")

    @search_server.tool()
    def web_search(destination: str, preference: str = "") -> dict[str, Any]:
        """返回本地美食方向与搜索提示。"""

        foods = LOCAL_FOODS.get(
            destination,
            [f"{destination}本地面食", f"{destination}街头小吃", f"{destination}特色家常菜"],
        )

        if "不要辣" in preference:
            search_tip = "你提到不要辣，点餐时可以优先选择清淡口味并备注少辣。"
        elif "美食" in preference:
            search_tip = "可以把热门小吃放在景点附近顺路解决，避免专门跨区觅食。"
        else:
            search_tip = "优先把吃饭安排在当天景点周边，减少来回折返。"

        return {
            "destination": destination,
            "foods": foods,
            "search_tip": search_tip,
        }

    services["search"] = MountedMcpService(
        name="search",
        tool_name="web_search",
        server=search_server,
        app=search_server.streamable_http_app(),
    )

    return services


def _stable_number(seed: str, low: int, high: int) -> int:
    """生成稳定的伪随机整数。

    用同一个 seed 多次调用，结果保持一致。
    这让学习项目的演示输出可重复，便于对比“缓存/反馈重规划”的效果。
    """

    span = high - low + 1
    return low + (sum(ord(char) for char in seed) % span)


def _season_summary(month: int) -> dict[str, str]:
    """按月份返回简化季节描述。"""

    if month in (12, 1, 2):
        return {
            "summary": "早晚偏冷，白天体感清爽",
            "temperature": "2-10C",
            "packing_tip": "建议带厚外套和保暖层。",
        }
    if month in (3, 4, 5):
        return {
            "summary": "整体温和，早晚略凉",
            "temperature": "14-24C",
            "packing_tip": "建议薄外套 + 舒适步行鞋。",
        }
    if month in (6, 7, 8):
        return {
            "summary": "偏热，午后体感较强",
            "temperature": "26-34C",
            "packing_tip": "建议轻薄衣物，记得补水和防晒。",
        }
    return {
        "summary": "白天舒适，夜间转凉",
        "temperature": "16-25C",
        "packing_tip": "建议长袖外搭，晚上加一层更稳妥。",
    }
