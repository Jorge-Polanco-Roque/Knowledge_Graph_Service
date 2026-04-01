#!/bin/bash
# Verify no secrets leaked into the Docker image.
# Usage: bash tests/integration/test_image_security.sh

set -e
IMAGE="${1:-neurograph-brain:latest}"
PASS=0
FAIL=0

check() {
    local desc="$1"
    local result="$2"
    if [ "$result" = "ok" ]; then
        echo "✅ $desc"
        PASS=$((PASS + 1))
    else
        echo "❌ FAIL: $desc — $result"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Image Security Checks: $IMAGE ==="
echo ""

# Check 1: No .env files
ENVFILES=$(docker run --rm "$IMAGE" find / -name ".env*" -not -name ".env.example" 2>/dev/null || true)
if [ -z "$ENVFILES" ]; then
    check "No .env files in image" "ok"
else
    check "No .env files in image" "found: $ENVFILES"
fi

# Check 2: No API keys in environment
KEYS=$(docker run --rm "$IMAGE" env 2>/dev/null | grep -iE "api_key|secret|password" || true)
if [ -z "$KEYS" ]; then
    check "No API keys baked into image" "ok"
else
    check "No API keys baked into image" "found env vars with sensitive names"
fi

# Check 3: Running as non-root
WHOAMI=$(docker run --rm "$IMAGE" whoami 2>/dev/null)
if [ "$WHOAMI" = "neurograph" ]; then
    check "Running as non-root user" "ok"
else
    check "Running as non-root user" "running as $WHOAMI"
fi

# Check 4: No git history
GIT=$(docker run --rm "$IMAGE" find / -name ".git" -type d 2>/dev/null || true)
if [ -z "$GIT" ]; then
    check "No .git directory" "ok"
else
    check "No .git directory" "found: $GIT"
fi

# Check 5: No pre-existing user data
DATA=$(docker run --rm "$IMAGE" ls /data/users/ 2>/dev/null || echo "")
if [ -z "$DATA" ]; then
    check "/data/users/ is empty" "ok"
else
    check "/data/users/ is empty" "found data: $DATA"
fi

# Check 6: No test files
TESTS=$(docker run --rm "$IMAGE" find /app -name "test_*" -o -name "*_test.py" 2>/dev/null || true)
if [ -z "$TESTS" ]; then
    check "No test files in image" "ok"
else
    check "No test files in image" "found: $TESTS"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] || exit 1
