"""FastAPI 入口。

这个模块刻意保持“薄”的 API 层，主要负责三件事：

1) 创建 FastAPI 应用并挂载静态前端（学习版单页）。
2) 把 6 个本地 Mock MCP 服务挂到 `/mcp/<service>` 下，方便你直接看到 MCP 端点形状。
3) 暴露最小的 3 个业务 API：
    - `POST /plan`：创建规划（可能返回 needs_input）
    - `POST /plan/{session_id}/resume`：补齐缺失信息后继续图执行
    - `POST /plan/{session_id}/feedback`：对已有攻略做“局部重规划”

学习主线建议配合阅读：
- `app/services/orchestrator.py`：API 与 LangGraph 的连接层
- `app/graph/workflow.py`：Plan -> Execute -> Replan 图
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models.api import FeedbackRequest, PlanRequest, ResumeRequest, TravelResponse
from app.services.orchestrator import TravelAssistantService


def create_app() -> FastAPI:
    """构建并返回 FastAPI app。

    说明：这里不做复杂的依赖注入框架，直接把 `TravelAssistantService`
    放进 `app.state`，让每个 endpoint 都能拿到同一套“学习版工作流组件”。
    """

    app = FastAPI(title=settings.app_name)
    app.state.service = TravelAssistantService()
    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"

    for service_name, mcp_app in app.state.service.mounted_mcp_apps().items():
        # 把每个 MCP 服务作为一个子应用挂载，方便你：
        # 1) 在浏览器里访问端点
        # 2) 理解“工具按领域服务组织”的边界
        app.mount(f"/mcp/{service_name}", mcp_app)
    app.state.service.attach_host_app(app)

    # 学习版前端是纯静态资源，因此直接使用 StaticFiles。
    # 真实项目里也可以替换为模板引擎或前后端分离部署。
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """返回学习版前端首页。

        这里不引模板引擎，直接回静态 HTML。
        因为页面不需要服务端拼接数据，越直接越容易学习。
        """

        return FileResponse(frontend_dir / "index.html")

    @app.post("/plan", response_model=TravelResponse)
    async def create_plan_endpoint(payload: PlanRequest, request: Request) -> TravelResponse:
        """创建一次新的旅游规划。

        关键点：真正的 Plan/Execute/Replan 不在这里，而在 `TravelAssistantService` 和 LangGraph 图中。
        API 层只负责把请求文本交给 service。
        """

        service: TravelAssistantService = request.app.state.service
        return await service.create_plan(payload.request)

    @app.post("/plan/{session_id}/resume", response_model=TravelResponse)
    async def resume_plan_endpoint(
        session_id: str,
        payload: ResumeRequest,
        request: Request,
    ) -> TravelResponse:
        """补充缺失信息并从 interrupt 位置继续执行图。"""

        service: TravelAssistantService = request.app.state.service
        try:
            return await service.resume_plan(session_id, payload.reply)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/plan/{session_id}/feedback", response_model=TravelResponse)
    async def feedback_endpoint(
        session_id: str,
        payload: FeedbackRequest,
        request: Request,
    ) -> TravelResponse:
        """对已有攻略提交局部反馈。

        这个接口体现学习项目的核心差异：
        - 不是“从头再生成一遍”
        - 而是先判断影响范围，再只重算受影响分区
        """

        service: TravelAssistantService = request.app.state.service
        try:
            return await service.apply_feedback(session_id, payload.feedback)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
