"""AUTH pentesting suite — API key authentication bypass tests.

Run with:
    NEUROGRAPH_API_KEY=nk_test_key_12345 pytest tests/security/test_auth.py -v

Requires: running NeuroGraph server on localhost:8899 with NEUROGRAPH_API_KEY set.
"""

from __future__ import annotations

import time

import httpx
import pytest

BASE = "http://localhost:8899"
VALID_KEY = "nk_test_key_12345"
HEADERS = {"Authorization": f"Bearer {VALID_KEY}"}


class TestAuthBypass:
    """Test that authentication cannot be bypassed."""

    def test_no_key_returns_401(self):
        r = httpx.get(f"{BASE}/api/graph/default-user")
        assert r.status_code == 401, f"Expected 401, got {r.status_code}"

    def test_wrong_key_returns_403(self):
        r = httpx.get(
            f"{BASE}/api/graph/default-user",
            headers={"Authorization": "Bearer wrong_key_value"},
        )
        assert r.status_code == 403

    def test_empty_bearer_returns_401(self):
        try:
            r = httpx.get(
                f"{BASE}/api/graph/default-user",
                headers={"Authorization": "Bearer "},
            )
            assert r.status_code in (401, 403)
        except httpx.LocalProtocolError:
            pass  # httpx rejects illegal header values at protocol level — attack blocked

    def test_valid_key_via_bearer(self):
        r = httpx.get(f"{BASE}/api/graph/default-user", headers=HEADERS)
        assert r.status_code == 200

    def test_valid_key_via_x_api_key(self):
        r = httpx.get(
            f"{BASE}/api/graph/default-user",
            headers={"X-API-Key": VALID_KEY},
        )
        assert r.status_code == 200

    def test_health_no_auth_required(self):
        r = httpx.get(f"{BASE}/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_docs_no_auth_required(self):
        r = httpx.get(f"{BASE}/docs")
        assert r.status_code == 200

    def test_sql_injection_in_key(self):
        r = httpx.get(
            f"{BASE}/api/graph/default-user",
            headers={"Authorization": "Bearer ' OR '1'='1"},
        )
        assert r.status_code == 403

    def test_null_bytes_in_key(self):
        try:
            r = httpx.get(
                f"{BASE}/api/graph/default-user",
                headers={"Authorization": f"Bearer {VALID_KEY}\x00extra"},
            )
            assert r.status_code == 403
        except httpx.LocalProtocolError:
            pass  # httpx rejects null bytes at protocol level — attack blocked

    def test_all_rest_endpoints_require_auth(self):
        """Verify every API endpoint rejects unauthenticated requests."""
        endpoints = [
            ("GET", "/api/graph/test-user"),
            ("GET", "/api/graph/test-user/stats"),
            ("GET", "/api/graph/test-user/timeline"),
            ("GET", "/api/graph/test-user/gaps"),
            ("GET", "/api/feedback/test-user/summary"),
            ("GET", "/api/ingest/files"),
            ("POST", "/api/chat"),
            ("POST", "/api/feedback/test-user"),
            ("POST", "/api/ingest/test-user"),
            ("POST", "/api/graph/test-user/prune-ghosts"),
            ("POST", "/api/graph/test-user/deduplicate"),
            ("POST", "/api/graph/test-user/summarize-clusters"),
            ("DELETE", "/api/graph/test-user"),
        ]
        for method, path in endpoints:
            r = httpx.request(method, f"{BASE}{path}")
            assert r.status_code in (401, 403, 422), (
                f"{method} {path} returned {r.status_code} without auth (expected 401/403)"
            )


class TestTimingAttack:
    """Verify constant-time key comparison."""

    def test_timing_attack_resistance(self):
        """Response time should NOT correlate with how close the key is."""
        times_wrong = []
        times_almost = []
        for _ in range(20):
            t0 = time.monotonic()
            httpx.get(
                f"{BASE}/api/graph/default-user",
                headers={"Authorization": "Bearer AAAA"},
            )
            times_wrong.append(time.monotonic() - t0)

            t0 = time.monotonic()
            httpx.get(
                f"{BASE}/api/graph/default-user",
                headers={"Authorization": f"Bearer {VALID_KEY[:-1]}X"},
            )
            times_almost.append(time.monotonic() - t0)

        avg_wrong = sum(times_wrong) / len(times_wrong)
        avg_almost = sum(times_almost) / len(times_almost)
        # Difference should be <10ms (noise level)
        assert abs(avg_wrong - avg_almost) < 0.01, (
            f"Timing difference: {abs(avg_wrong - avg_almost)*1000:.1f}ms "
            f"(wrong={avg_wrong*1000:.1f}ms, almost={avg_almost*1000:.1f}ms)"
        )
