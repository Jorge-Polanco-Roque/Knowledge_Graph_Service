#!/bin/bash
# Verify data persists across container restarts.
# Usage: bash tests/integration/test_persistence.sh
# Requires: running docker-compose stack, valid API keys.

set -e
API="http://localhost:8899"
KEY="Authorization: Bearer nk_test_key_12345"
USER="persist-test"

echo "=== Data Persistence Test ==="
echo ""

echo "Step 1: Reset user..."
curl -sf -X DELETE "$API/api/graph/$USER" -H "$KEY" | jq -r .status

echo "Step 2: Send a chat message..."
RESP=$(curl -sf -X POST "$API/api/chat" \
  -H "$KEY" -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER\", \"message\": \"Neural networks use backpropagation for learning.\"}" \
  --max-time 60)
echo "Response: $(echo "$RESP" | jq -r .response | head -c 100)..."

echo "Step 3: Check node count BEFORE restart..."
BEFORE=$(curl -sf "$API/api/graph/$USER/stats" -H "$KEY" | jq -r .node_count)
echo "Nodes before: $BEFORE"

if [ "$BEFORE" -eq 0 ] || [ -z "$BEFORE" ]; then
    echo "❌ FAIL: No nodes created after chat"
    exit 1
fi

echo "Step 4: Restart container..."
docker compose restart
echo "Waiting for container to be healthy..."
for i in $(seq 1 20); do
    if curl -sf "$API/health" > /dev/null 2>&1; then
        echo "Container ready after ${i}s"
        break
    fi
    sleep 1
done

echo "Step 5: Check node count AFTER restart..."
AFTER=$(curl -sf "$API/api/graph/$USER/stats" -H "$KEY" | jq -r .node_count)
echo "Nodes after:  $AFTER"

if [ "$BEFORE" -eq "$AFTER" ] && [ "$BEFORE" -gt 0 ]; then
    echo ""
    echo "✅ PASS: Data persisted across restart ($BEFORE nodes)"
else
    echo ""
    echo "❌ FAIL: Data lost! Before=$BEFORE After=$AFTER"
    exit 1
fi

# Cleanup
echo ""
echo "Cleanup: resetting test user..."
curl -sf -X DELETE "$API/api/graph/$USER" -H "$KEY" | jq -r .status
echo "Done."
