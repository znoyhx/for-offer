"""主链路测试。"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from app.main import create_app


def extract_section(markdown: str, title: str) -> str:
    pattern = rf"## {re.escape(title)}\n(?P<body>.*?)(?=\n## |\Z)"
    match = re.search(pattern, markdown, flags=re.S)
    assert match is not None, f"missing section: {title}"
    return match.group("body").strip()


def test_homepage_is_served() -> None:
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "AI Travel Assistant" in response.text
    assert "/static/app.js" in response.text


def test_static_assets_are_served() -> None:
    client = TestClient(create_app())

    response = client.get("/static/app.css")

    assert response.status_code == 200
    assert ".page-shell" in response.text


def test_plan_interrupts_when_dates_are_missing() -> None:
    client = TestClient(create_app())

    response = client.post("/plan", json={"request": "我想从北京去上海旅游"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "needs_input"
    assert "出发日期" in payload["question"]
    assert "返程日期" in payload["question"]


def test_resume_generates_complete_guide() -> None:
    client = TestClient(create_app())

    first = client.post("/plan", json={"request": "我想从北京去上海旅游"})
    session_id = first.json()["session_id"]

    second = client.post(
        f"/plan/{session_id}/resume",
        json={"reply": "2026-05-01出发，2026-05-04返回"},
    )

    assert second.status_code == 200
    payload = second.json()
    assert payload["status"] == "completed"
    guide = payload["guide"]

    for title in ("交通", "住宿", "天气", "景点", "美食"):
        assert f"## {title}" in guide

    assert "G" in guide
    assert "MU" in guide or "CZ" in guide
    assert "建议薄外套" in guide


def test_feedback_only_rewrites_the_hotel_section() -> None:
    client = TestClient(create_app())

    first = client.post("/plan", json={"request": "我想从北京去上海旅游"})
    session_id = first.json()["session_id"]
    second = client.post(
        f"/plan/{session_id}/resume",
        json={"reply": "2026-05-01出发，2026-05-04返回"},
    )
    original_guide = second.json()["guide"]

    third = client.post(
        f"/plan/{session_id}/feedback",
        json={"feedback": "住宿想更便宜一点，其他不动"},
    )

    assert third.status_code == 200
    updated_guide = third.json()["guide"]

    assert "预算友好" in extract_section(updated_guide, "住宿")
    assert extract_section(original_guide, "天气") == extract_section(updated_guide, "天气")
    assert extract_section(original_guide, "景点") == extract_section(updated_guide, "景点")
    assert extract_section(original_guide, "美食") == extract_section(updated_guide, "美食")
