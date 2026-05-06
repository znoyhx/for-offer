"""工具结果缓存。

这里不引入向量库，改成一个足够透明的“RAG 风格”最小实现：
- 先做完全相同查询命中。
- 再做 token overlap 相似度检索。

这样既能学到“先检索历史结果再复用”的思路，又不会把项目复杂度抬太高。
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CacheEntry:
    """缓存条目。

    - key：用于“完全相同查询”命中
    - query_text：用于“相似查询”命中（Jaccard token overlap）
    - result：结构化工具结果（会 deepcopy，避免外部改写污染缓存）
    """

    service: str
    tool_name: str
    key: str
    query_text: str
    result: dict[str, Any]


class ToolResultCache:
    """内存缓存（最小可解释实现）。

    命中策略分两层：
    1) 完全相同查询：service + tool + arguments 的稳定序列化 key
    2) 相似查询：把参数压成 query_text，做 Jaccard 相似度检索

    这种实现足够“透明”，非常适合学习：你能清楚看到它为什么命中/没命中。
    """

    def __init__(self, similarity_threshold: float) -> None:
        self._similarity_threshold = similarity_threshold
        self._exact_index: dict[str, CacheEntry] = {}
        self._entries: list[CacheEntry] = []

    def lookup(self, service: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """查询缓存。

        返回的是 deepcopy 后的结果，避免调用方修改返回值导致缓存被污染。
        """

        key = self._build_key(service, tool_name, arguments)
        exact_hit = self._exact_index.get(key)
        if exact_hit:
            return copy.deepcopy(exact_hit.result)

        query_text = self._build_query_text(service, tool_name, arguments)
        best_score = 0.0
        best_entry: CacheEntry | None = None

        for entry in self._entries:
            if entry.service != service or entry.tool_name != tool_name:
                continue
            score = self._jaccard_similarity(query_text, entry.query_text)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self._similarity_threshold:
            return copy.deepcopy(best_entry.result)

        return None

    def store(self, service: str, tool_name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        """写入缓存（同时更新 exact 索引与 entry 列表）。"""

        key = self._build_key(service, tool_name, arguments)
        entry = CacheEntry(
            service=service,
            tool_name=tool_name,
            key=key,
            query_text=self._build_query_text(service, tool_name, arguments),
            result=copy.deepcopy(result),
        )
        self._exact_index[key] = entry
        self._entries.append(entry)

    def _build_key(self, service: str, tool_name: str, arguments: dict[str, Any]) -> str:
        # sort_keys=True 保证 dict 序列化稳定，从而让“完全相同查询”真的能命中。
        normalised = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        return f"{service}:{tool_name}:{normalised}"

    def _build_query_text(self, service: str, tool_name: str, arguments: dict[str, Any]) -> str:
        # 这里用“键:值”的扁平文本，配合 token overlap 做最小相似检索。
        flattened_arguments = " ".join(f"{key}:{value}" for key, value in sorted(arguments.items()) if value)
        return f"{service} {tool_name} {flattened_arguments}".lower()

    def _jaccard_similarity(self, left: str, right: str) -> float:
        # 纯集合相似度：实现简单、可解释，足以支持学习版的“相似命中”。
        left_tokens = self._tokenise(left)
        right_tokens = self._tokenise(right)

        if not left_tokens or not right_tokens:
            return 0.0

        intersection = left_tokens & right_tokens
        union = left_tokens | right_tokens
        return len(intersection) / len(union)

    def _tokenise(self, text: str) -> set[str]:
        # 同时保留英文/数字 token 与中文单字：
        # - 英文部分能区分 service/tool/参数名
        # - 中文单字能覆盖“便宜/安静/少走路”等偏好
        ascii_tokens = set(re.findall(r"[a-z0-9_-]+", text.lower()))
        chinese_chars = set(re.findall(r"[\u4e00-\u9fff]", text))
        return ascii_tokens | chinese_chars

