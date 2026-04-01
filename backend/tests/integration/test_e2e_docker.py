"""End-to-end tests for the NeuroGraph Brain API.

Run with:
    NEUROGRAPH_API_KEY=nk_test_key_12345 pytest tests/integration/test_e2e_docker.py -v --timeout=120

Requires: running NeuroGraph server on localhost:8899 with valid OPENAI_API_KEY.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

BASE = "http://localhost:8899"
USER = "e2e-test-user"
HEADERS = {"Authorization": "Bearer nk_test_key_12345", "Content-Type": "application/json"}


class TestHealthAndBasics:
    """Basic health and connectivity tests."""

    def test_health(self):
        r = httpx.get(f"{BASE}/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_openapi_docs(self):
        r = httpx.get(f"{BASE}/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert "paths" in schema
        assert "/api/chat" in schema["paths"]


class TestPipelineE2E:
    """Full agent pipeline: chat -> concepts -> graph update."""

    @classmethod
    def setup_class(cls):
        """Reset test user before all tests."""
        httpx.delete(f"{BASE}/api/graph/{USER}", headers=HEADERS, timeout=10)

    def test_01_empty_graph(self):
        r = httpx.get(f"{BASE}/api/graph/{USER}", headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert len(data.get("nodes", [])) == 0

    def test_02_chat_creates_concepts(self):
        """A chat message should extract concepts and update the graph."""
        r = httpx.post(
            f"{BASE}/api/chat",
            headers=HEADERS,
            json={
                "user_id": USER,
                "message": "Fairness in machine learning hiring algorithms is important for society.",
            },
            timeout=60,
        )
        assert r.status_code == 200
        data = r.json()
        assert "response" in data
        assert len(data["response"]) > 10

    def test_03_graph_has_nodes(self):
        """After chat, graph should have nodes."""
        r = httpx.get(f"{BASE}/api/graph/{USER}", headers=HEADERS)
        data = r.json()
        nodes = data.get("nodes", [])
        assert len(nodes) > 0
        labels = [n["label"].lower() for n in nodes]
        assert any("fairness" in l or "hiring" in l or "machine" in l for l in labels)

    def test_04_graph_stats(self):
        r = httpx.get(f"{BASE}/api/graph/{USER}/stats", headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert data.get("total_nodes", data.get("node_count", 0)) > 0

    def test_05_feedback_submission(self):
        r = httpx.post(
            f"{BASE}/api/feedback/{USER}",
            headers=HEADERS,
            json={
                "message_id": "test_msg_001",
                "rating": -1,
                "comment": "The response was inaccurate about hiring.",
                "query": "What is fairness?",
                "response": "test response",
                "timestamp": "2026-03-31T00:00:00Z",
            },
        )
        assert r.status_code == 200

    def test_06_feedback_summary(self):
        r = httpx.get(f"{BASE}/api/feedback/{USER}/summary", headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1

    def test_07_knowledge_gaps(self):
        r = httpx.get(f"{BASE}/api/graph/{USER}/gaps", headers=HEADERS)
        assert r.status_code == 200
        assert "total_nodes" in r.json()

    def test_08_graph_reset(self):
        r = httpx.delete(f"{BASE}/api/graph/{USER}", headers=HEADERS)
        assert r.status_code == 200
        # Verify empty
        r2 = httpx.get(f"{BASE}/api/graph/{USER}", headers=HEADERS)
        assert len(r2.json().get("nodes", [])) == 0


class TestIngestionE2E:
    """File ingestion pipeline."""

    def test_list_files(self):
        r = httpx.get(f"{BASE}/api/ingest/files", headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert "total_files" in data


class TestWebSocketE2E:
    """WebSocket streaming pipeline."""

    @pytest.mark.asyncio
    async def test_streaming_chat(self):
        try:
            import websockets
        except ImportError:
            pytest.skip("websockets package required for WS tests")

        uri = f"ws://localhost:8899/ws/{USER}?token=nk_test_key_12345"
        async with websockets.connect(uri, close_timeout=30) as ws:
            await ws.send(json.dumps({
                "type": "user_message",
                "data": {"content": "What is demographic parity?"},
            }))

            chunks: list[str] = []
            got_end = False
            deadline = time.monotonic() + 45

            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=15)
                    msg = json.loads(raw)

                    if msg["type"] == "response_chunk":
                        chunks.append(msg["data"]["content"])
                    elif msg["type"] == "response_end":
                        got_end = True
                        break
                    elif msg["type"] == "error":
                        pytest.fail(f"WS error: {msg['data']}")
                except asyncio.TimeoutError:
                    break

            assert len(chunks) > 0, "No streaming chunks received"
            assert got_end, "Never received response_end"
            full_response = "".join(chunks)
            assert len(full_response) > 20
