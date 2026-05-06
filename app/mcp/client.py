"""MCP 客户端包装。

这个学习项目保留了两层能力：
1. 对外：FastAPI 会挂出真正的 Streamable HTTP MCP 端点。
2. 对内：为了让离线演示稳定可跑，工作流直接调用 `FastMCP.call_tool()`。

这样既能学习 MCP 服务的定义方式，也不会因为本地子应用 lifespan 管理
把样例复杂度抬得过高。
"""

from __future__ import annotations

from typing import Any

from app.mcp.mock_servers import MountedMcpService


class McpClientManager:
    """把“如何连 MCP 服务”封装起来，业务层只关心 service name。

    学习版的核心目的：
    - 让 `TravelWorkerAgent` 在不关心协议细节的前提下调用工具
    - 同时保留 MCP 服务的真实形状（可以挂到 FastAPI 上作为子应用）
    """

    def __init__(self, services: dict[str, MountedMcpService]) -> None:
        self._services = services

    def attach_host_app(self, app: Any) -> None:
        """保留这个接口，方便主应用在创建时显式表达“这些 MCP 服务属于我”。

        当前学习版内部直接调 `FastMCP.call_tool()`，所以这里不需要额外逻辑；
        但接口保留后，后续想切回 HTTP/stdio 客户端也不需要改应用层。
        """

        _ = app

    async def call_tool(self, service_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """调用某个 MCP 服务的默认工具，并返回结构化 dict。

        FastMCP 的 `call_tool` 返回值可能是：
        - 直接 dict（结构化结果）
        - (content_blocks, structured_result)
        - 其它可迭代内容块（只有 text）

        学习版优先取结构化结果，以便父 Agent 合并并渲染攻略。
        """

        service = self._services[service_name]
        result = await service.server.call_tool(service.tool_name, arguments)

        if isinstance(result, dict):
            return result

        # FastMCP 在 convert_result=True 时，常见返回是：
        # (content_blocks, structured_result)
        # 学习项目更需要第二个结构化结果，方便后续父 Agent 合并攻略。
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return dict(result[1])

        text_parts: list[str] = []
        blocks = result[0] if isinstance(result, tuple) and result else result
        for block in blocks:
            text = getattr(block, "text", None)
            if text:
                text_parts.append(text)
        return {"text": "\n".join(text_parts)}

    def mounted_apps(self) -> dict[str, Any]:
        """返回可以挂到 FastAPI 的子应用对象集合。"""

        return {name: service.app for name, service in self._services.items()}
