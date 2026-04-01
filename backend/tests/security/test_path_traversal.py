"""PATH TRAVERSAL pentesting suite.

Run with:
    NEUROGRAPH_API_KEY=nk_test_key_12345 pytest tests/security/test_path_traversal.py -v

Requires: running NeuroGraph server on localhost:8899.
"""

from __future__ import annotations

import httpx
import pytest

BASE = "http://localhost:8899"
HEADERS = {"Authorization": "Bearer nk_test_key_12345", "Content-Type": "application/json"}


class TestPathTraversal:
    """Test that path traversal attacks are blocked."""

    @pytest.mark.parametrize("filename", [
        "../../../etc/passwd",
        "..%2F..%2F..%2Fetc%2Fpasswd",
        "....//....//etc/passwd",
        "/etc/passwd",
        "subfolder/../../../etc/passwd",
        "..\\..\\..\\etc\\passwd",
        ".env",
        "../.env",
        "../../backend/.env",
    ])
    def test_filename_traversal_blocked(self, filename: str):
        r = httpx.post(f"{BASE}/api/ingest/default-user/{filename}", headers=HEADERS)
        assert r.status_code in (400, 403, 404, 422), (
            f"Traversal not blocked for filename '{filename}': got {r.status_code}"
        )

    @pytest.mark.parametrize("user_id", [
        "../../etc",
        "../admin",
        "user/../../etc",
        "user%2F..%2F..%2Fetc",
        "a" * 65,  # exceeds max length
        "user<script>",
        "user;rm -rf",
        "",
    ])
    def test_user_id_traversal_blocked(self, user_id: str):
        r = httpx.get(f"{BASE}/api/graph/{user_id}", headers=HEADERS)
        assert r.status_code in (400, 404, 422), (
            f"User ID traversal not blocked for '{user_id}': got {r.status_code}"
        )

    def test_valid_filename_accepted(self):
        """A properly formatted filename should not be rejected by validation."""
        r = httpx.post(f"{BASE}/api/ingest/default-user/article_1.docx", headers=HEADERS)
        # 404 is fine (file may not exist), but NOT 400/403
        assert r.status_code in (200, 404), (
            f"Valid filename rejected: got {r.status_code}"
        )

    def test_valid_user_id_accepted(self):
        r = httpx.get(f"{BASE}/api/graph/test-valid-user", headers=HEADERS)
        assert r.status_code == 200
