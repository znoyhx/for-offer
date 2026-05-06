"""核心业务工具包。

这里放的是“与工作流强相关但与 API 无关”的能力：
- cache：工具结果缓存与相似命中
- context：上下文压缩（保留 ToolMessage + 最近 N 条消息）
- llm：父 Agent 的教学版实现（Plan/Replan/Feedback 逻辑）
- parsing：需求解析与缺失字段判断
"""

