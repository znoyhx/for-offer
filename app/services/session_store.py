"""最小会话存储。

草稿里没有提数据库，因此这里故意使用内存版会话存储，
把学习重点放在工作流和 Agent 协作上。
"""

from __future__ import annotations

from copy import deepcopy

from langchain_core.messages import messages_from_dict, messages_to_dict

from app.models.state import TravelAssistantState


class SessionStore:
    """最小内存会话存储。

    目的：支撑三类交互都能“接着上次的状态继续走”：
    - create：创建会话
    - resume：从 interrupt 处继续
    - feedback：基于上一次 completed 的状态做增量重规划

    这里需要对 messages 做序列化/反序列化：LangChain 的消息对象
    不能直接 JSON 化，但可以用 `messages_to_dict/messages_from_dict` 转换。
    """

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, object]] = {}

    def create(self, session_id: str, request_text: str) -> None:
        """初始化一条新会话记录。"""

        self._sessions[session_id] = {
            "request_text": request_text,
            "state": None,
            "revision": 0,
        }

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def save_state(self, session_id: str, state: TravelAssistantState) -> None:
        """保存图的最新状态（通常是 completed 时）。"""

        self._assert_exists(session_id)
        self._sessions[session_id]["state"] = self._serialise_state(state)

    def load_state(self, session_id: str) -> TravelAssistantState | None:
        """读取并还原状态对象。"""

        self._assert_exists(session_id)
        stored_state = self._sessions[session_id]["state"]
        if stored_state is None:
            return None
        return self._deserialise_state(stored_state)

    def next_revision(self, session_id: str) -> int:
        """返回并递增反馈修订号，用于生成新的 feedback thread_id。"""

        self._assert_exists(session_id)
        revision = int(self._sessions[session_id]["revision"]) + 1
        self._sessions[session_id]["revision"] = revision
        return revision

    def _serialise_state(self, state: TravelAssistantState) -> dict[str, object]:
        """把 state 转成可以存储的 dict（messages -> dict）。"""

        payload = deepcopy(dict(state))
        if "messages" in payload:
            payload["messages"] = messages_to_dict(payload["messages"])
        return payload

    def _deserialise_state(self, payload: dict[str, object]) -> TravelAssistantState:
        """把存储态 dict 还原成运行态 state（dict -> messages）。"""

        state = deepcopy(payload)
        if "messages" in state:
            state["messages"] = messages_from_dict(state["messages"])
        return state

    def _assert_exists(self, session_id: str) -> None:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown session id: {session_id}")

