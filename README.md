# AI Travel Assistant Learning Demo

这是一个按 `draft.txt` 里的技术栈和功能边界复现出来的学习项目。

现在它包含两部分：

- 后端：`FastAPI + LangGraph + LangChain + FastMCP`
- 前端：一个不引入额外框架的静态单页，方便你直接操作完整链路

这个项目只实现这些能力：

- 创建旅游规划
- 信息不足时暂停，并让用户补充
- 生成住宿 / 交通 / 天气 / 景点 / 美食攻略
- 对已有攻略做局部反馈重规划
- 缓存复用、上下文压缩、多 Agent 分工

这个项目刻意不实现这些东西：

- 登录
- 数据库
- 推荐系统
- 地图可视化
- 真实第三方 API 接入

这样做的目的只有一个：把多 Agent 工作流本身做得够清楚、够容易学。

## Run

如果你的工作区里已经有 `.packages`，直接运行：

```bash
python run.py
```

如果你是从头拿到这个项目，先安装依赖到项目本地目录：

```bash
pip install --target .packages -r requirements.txt
python run.py
```

服务默认启动在 `http://127.0.0.1:8000`。

## Frontend

启动后直接打开：

```text
http://127.0.0.1:8000/
```

页面只提供 3 个交互动作：

1. 提交旅游需求
2. 补充后端要求的信息
3. 对攻略提交局部反馈

## API

你也可以继续直接调接口。

1. 创建旅游规划

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/plan `
  -ContentType 'application/json' `
  -Body '{"request":"我想从北京去上海旅游"}'
```

2. 如果返回 `needs_input`，继续补充信息

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/plan/<session_id>/resume `
  -ContentType 'application/json' `
  -Body '{"reply":"2026-05-01出发，2026-05-04返回"}'
```

3. 对已有攻略提反馈

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/plan/<session_id>/feedback `
  -ContentType 'application/json' `
  -Body '{"feedback":"住宿想更便宜一点，其他不动"}'
```

## MCP Endpoints

为了方便学习，6 个 MCP 服务仍然挂了出来：

- `/mcp/map/mcp`
- `/mcp/train/mcp`
- `/mcp/flight/mcp`
- `/mcp/weather/mcp`
- `/mcp/hotel/mcp`
- `/mcp/search/mcp`

这个学习版内部为了保证离线稳定可跑，工作流直接调用 `FastMCP.call_tool()`；
但这些 HTTP 端点仍然保留，方便你观察 MCP 服务是怎么定义和挂载的。

## Test

```bash
python -m pytest -q
```

## Learn More

更详细的讲解看 [docs/学习项目文档.md](docs/学习项目文档.md)。

