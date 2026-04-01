"""WebSocket handler for real-time NeuroGraph interaction with streaming responses."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import traceback

from fastapi import WebSocket, WebSocketDisconnect
from langchain_openai import ChatOpenAI

from .auth import validate_ws_token
from .context import current_openai_key

# --- Security: Prompt injection detection ---
_INJECTION_PATTERNS = [
    "ignore all previous", "ignore your instructions", "you are now",
    "act as if", "pretend to be", "new persona", "forget your",
    "reveal your prompt", "show me your system", "system prompt",
    "override your", "disregard your", "ignore the above",
    "do not follow", "bypass your", "jailbreak",
]
_VALID_USER_ID = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')

# --- Connection & rate limits ---
MAX_CONNECTIONS_PER_USER = 3
MAX_MESSAGE_LENGTH = 10_000
MIN_MESSAGE_INTERVAL = 1.0  # seconds

_active_connections: dict[str, int] = {}
_last_message_time: dict[str, float] = {}


def _check_injection(content: str) -> bool:
    lower = content.lower()
    return any(p in lower for p in _INJECTION_PATTERNS)

from ..agent.nodes import (
    SYSTEM_PROMPT,
    _get_memory_manager,
    _grounding_check,
    activate_memory,
    build_context,
    classify_query,
    extract_concepts,
    retrieve_documents,
    update_graph,
)
from ..config import settings
from ..ingestion.ingest import save_conversation
from ..models.messages import (
    AgentResponseData,
    ChatMessage,
    MessageRole,
    WSMessage,
    WSMessageType,
)

logger = logging.getLogger(__name__)


async def websocket_endpoint(websocket: WebSocket, user_id: str) -> None:
    """Handle a WebSocket connection for a given user.

    Protocol:
        Client sends: {"type": "user_message", "data": {"content": "..."}}
        Server sends (streaming):
            1. {"type": "response_chunk", "data": {"content": "token..."}}  (many)
            2. {"type": "response_end", "data": {...concepts, graph_diff...}}
            3. {"type": "graph_diff", "data": {...}}
            4. {"type": "spreading_activation", "data": {"waves": [...]}}
    """
    # Validate user_id
    if not _VALID_USER_ID.match(user_id):
        await websocket.close(code=4000, reason="Invalid user_id")
        return

    # --- Auth: validate API key ---
    if not validate_ws_token(websocket):
        await websocket.close(code=4003, reason="Invalid or missing API key")
        return

    # --- Connection limit per user ---
    count = _active_connections.get(user_id, 0)
    if count >= MAX_CONNECTIONS_PER_USER:
        await websocket.close(code=4029, reason="Too many connections")
        return

    await websocket.accept()
    _active_connections[user_id] = count + 1
    logger.info("WebSocket connected: user_id=%s (connections: %d)", user_id, count + 1)

    # Resolve the OpenAI key for this connection
    # WS clients can pass it as a query param: ?openai_key=sk-...
    ws_openai_key = websocket.query_params.get("openai_key", "").strip()
    if not ws_openai_key:
        ws_openai_key = settings.openai_api_key

    conversation: list[ChatMessage] = []

    try:
        while True:
            raw = await websocket.receive_json()

            msg_type = raw.get("type", "")
            if msg_type != WSMessageType.USER_MESSAGE:
                await websocket.send_json(
                    WSMessage(
                        type=WSMessageType.ERROR,
                        data={"message": f"Unknown message type: {msg_type}"},
                    ).model_dump(mode="json")
                )
                continue

            content = raw.get("data", {}).get("content", "").strip()

            # --- Message length limit ---
            if len(content) > MAX_MESSAGE_LENGTH:
                await websocket.send_json(
                    WSMessage(
                        type=WSMessageType.ERROR,
                        data={"message": f"Message too long. Max {MAX_MESSAGE_LENGTH} characters."},
                    ).model_dump(mode="json")
                )
                continue

            # --- Message rate limit ---
            now = time.monotonic()
            last = _last_message_time.get(user_id, 0)
            if now - last < MIN_MESSAGE_INTERVAL:
                await websocket.send_json(
                    WSMessage(
                        type=WSMessageType.ERROR,
                        data={"message": "Rate limit: wait 1 second between messages."},
                    ).model_dump(mode="json")
                )
                continue
            _last_message_time[user_id] = now

            # Prompt injection detection — don't block, but flag for the LLM
            if _check_injection(content):
                logger.warning("Potential prompt injection from user %s: %s", user_id, content[:100])

            if not content:
                await websocket.send_json(
                    WSMessage(
                        type=WSMessageType.ERROR,
                        data={"message": "Empty message content"},
                    ).model_dump(mode="json")
                )
                continue

            # Add user message to history
            user_msg = ChatMessage(role=MessageRole.USER, content=content)
            conversation.append(user_msg)

            # Run the streaming agent pipeline
            try:
                # Set the OpenAI key for this request scope
                token = current_openai_key.set(ws_openai_key)
                try:
                    result = await _run_agent_streaming(
                        user_id, list(conversation), websocket
                    )
                finally:
                    current_openai_key.reset(token)
            except Exception as exc:
                logger.error("Agent error: %s", exc, exc_info=True)
                await websocket.send_json(
                    WSMessage(
                        type=WSMessageType.ERROR,
                        data={"message": "An error occurred processing your message."},
                    ).model_dump(mode="json")
                )
                continue

            # Add assistant response to history
            assistant_msg = ChatMessage(
                role=MessageRole.ASSISTANT, content=result["response"]
            )
            conversation.append(assistant_msg)

            # Send response_end with concepts and diff
            response_data = AgentResponseData(
                content=result["response"],
                extracted_concepts=result.get("extracted_concepts", []),
                graph_diff=result.get("graph_diff"),
            )
            await websocket.send_json(
                WSMessage(
                    type=WSMessageType.RESPONSE_END,
                    data=response_data.model_dump(mode="json"),
                ).model_dump(mode="json")
            )

            # Send graph diff
            graph_diff = result.get("graph_diff")
            if graph_diff:
                await websocket.send_json(
                    WSMessage(
                        type=WSMessageType.GRAPH_DIFF,
                        data=graph_diff.model_dump(mode="json"),
                    ).model_dump(mode="json")
                )

            # Send spreading activation waves
            waves = result.get("spreading_waves", [])
            if waves:
                await websocket.send_json(
                    WSMessage(
                        type=WSMessageType.SPREADING_ACTIVATION,
                        data={"waves": waves},
                    ).model_dump(mode="json")
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: user_id=%s", user_id)
    except Exception:
        logger.exception("WebSocket error for user_id=%s", user_id)
    finally:
        _active_connections[user_id] = max(0, _active_connections.get(user_id, 1) - 1)
        if len(conversation) >= 2:
            try:
                msgs = [{"role": m.role.value, "content": m.content} for m in conversation]
                save_conversation(user_id, msgs)
            except Exception:
                logger.exception("Failed to save conversation for %s", user_id)


async def _run_agent_streaming(
    user_id: str, messages: list[ChatMessage], websocket: WebSocket
) -> dict:
    """Run agent pipeline with token-level streaming for the generation step.

    Phase 1: Pre-generation (sync nodes run in executor) — ~1-2s
    Phase 2: Stream LLM response token-by-token via WebSocket — ~3-8s
    Phase 3: Update graph (sync, run in executor)
    """
    state: dict = {"messages": messages, "user_id": user_id}
    loop = asyncio.get_event_loop()

    # Query routing: classify to determine pipeline depth
    user_content = messages[-1].content if messages else ""
    route = await loop.run_in_executor(None, classify_query, user_content)
    logger.info("Query route: %s for '%s'", route, user_content[:50])

    if route == "greeting":
        # Skip the entire pipeline for greetings
        state["graph_context"] = ""
        state["extracted_concepts"] = []
        state["document_chunks"] = []
    elif route == "simple":
        # RAG only, skip graph activation (faster)
        state.update(await loop.run_in_executor(None, extract_concepts, state))
        state.update(await loop.run_in_executor(None, retrieve_documents, state))
        state.update(await loop.run_in_executor(None, build_context, state))
    else:
        # Full pipeline for complex queries
        state.update(await loop.run_in_executor(None, extract_concepts, state))
        state.update(await loop.run_in_executor(None, activate_memory, state))
        state.update(await loop.run_in_executor(None, retrieve_documents, state))
        state.update(await loop.run_in_executor(None, build_context, state))

    # Phase 2: Stream LLM response
    graph_context = state.get("graph_context", "")
    system = SYSTEM_PROMPT.format(
        graph_context=graph_context if graph_context else "No relevant context yet."
    )

    llm_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for msg in messages:
        llm_messages.append({"role": msg.role.value, "content": msg.content})

    # Use the per-request OpenAI key from contextvar
    api_key = current_openai_key.get(settings.openai_api_key)
    if not api_key:
        raise ValueError("No OpenAI API key configured. Provide via X-OpenAI-Key header or OPENAI_API_KEY env var.")

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=api_key,
        temperature=0.3,
        streaming=True,
    )

    full_response = ""
    async for chunk in llm.astream(llm_messages):
        if chunk.content:
            full_response += chunk.content
            try:
                await websocket.send_json({
                    "type": WSMessageType.RESPONSE_CHUNK.value,
                    "data": {"content": chunk.content},
                })
            except Exception:
                break  # WebSocket disconnected mid-stream

    # Post-processing: grounding check — append disclaimer if unverified claims found
    full_response = _grounding_check(full_response, graph_context)
    if full_response.endswith("*") and "\n\n---\n" in full_response:
        # Grounding disclaimer was added — send it as final chunk
        disclaimer = full_response[full_response.rindex("\n\n---\n"):]
        try:
            await websocket.send_json({
                "type": WSMessageType.RESPONSE_CHUNK.value,
                "data": {"content": disclaimer},
            })
        except Exception:
            pass

    state["response"] = full_response

    # Phase 3: Update graph
    state.update(await loop.run_in_executor(None, update_graph, state))

    return state
