"""WebSocket handler for real-time NeuroGraph interaction with streaming responses."""

from __future__ import annotations

import asyncio
import logging
import re
import traceback

from fastapi import WebSocket, WebSocketDisconnect
from langchain_openai import ChatOpenAI

# --- Security: Prompt injection detection ---
_INJECTION_PATTERNS = [
    "ignore all previous", "ignore your instructions", "you are now",
    "act as if", "pretend to be", "new persona", "forget your",
    "reveal your prompt", "show me your system", "system prompt",
    "override your", "disregard your", "ignore the above",
    "do not follow", "bypass your", "jailbreak",
]
_VALID_USER_ID = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')


def _check_injection(content: str) -> bool:
    lower = content.lower()
    return any(p in lower for p in _INJECTION_PATTERNS)

from ..agent.nodes import (
    SYSTEM_PROMPT,
    _get_memory_manager,
    _grounding_check,
    activate_memory,
    build_context,
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

    await websocket.accept()
    logger.info("WebSocket connected: user_id=%s", user_id)

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
                result = await _run_agent_streaming(
                    user_id, list(conversation), websocket
                )
            except Exception as exc:
                logger.error("Agent error: %s\n%s", exc, traceback.format_exc())
                await websocket.send_json(
                    WSMessage(
                        type=WSMessageType.ERROR,
                        data={"message": f"Agent error: {str(exc)}"},
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
    except Exception as exc:
        logger.error(
            "WebSocket error for user_id=%s: %s\n%s",
            user_id, exc, traceback.format_exc(),
        )
    finally:
        if len(conversation) >= 2:
            try:
                msgs = [{"role": m.role.value, "content": m.content} for m in conversation]
                save_conversation(user_id, msgs)
            except Exception as exc:
                logger.error("Failed to save conversation: %s", exc)


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

    # Phase 1: Pre-generation pipeline
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

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
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
