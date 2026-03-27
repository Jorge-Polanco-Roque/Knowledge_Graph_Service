"""WebSocket handler for real-time NeuroGraph interaction."""

from __future__ import annotations

import logging
import traceback

from fastapi import WebSocket, WebSocketDisconnect

from ..agent.graph_agent import create_agent
from ..ingestion.ingest import save_conversation
from ..models.messages import (
    AgentResponseData,
    ChatMessage,
    MessageRole,
    WSMessage,
    WSMessageType,
)

logger = logging.getLogger(__name__)

# Compile the agent once at module level
_agent = create_agent()


async def websocket_endpoint(websocket: WebSocket, user_id: str) -> None:
    """Handle a WebSocket connection for a given user.

    Protocol:
        Client sends: {"type": "user_message", "data": {"content": "..."}}
        Server sends:
            1. {"type": "agent_response", "data": {...}}
            2. {"type": "graph_diff", "data": {...}}
            3. {"type": "spreading_activation", "data": {"waves": [...]}}
    """
    await websocket.accept()
    logger.info("WebSocket connected: user_id=%s", user_id)

    # Maintain conversation history for this connection
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

            # Run the agent pipeline
            try:
                result = await _run_agent(user_id, list(conversation))
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

            # 1. Send agent response
            response_data = AgentResponseData(
                content=result["response"],
                extracted_concepts=result.get("extracted_concepts", []),
                graph_diff=result.get("graph_diff"),
            )
            await websocket.send_json(
                WSMessage(
                    type=WSMessageType.AGENT_RESPONSE,
                    data=response_data.model_dump(mode="json"),
                ).model_dump(mode="json")
            )

            # 2. Send graph diff
            graph_diff = result.get("graph_diff")
            if graph_diff:
                await websocket.send_json(
                    WSMessage(
                        type=WSMessageType.GRAPH_DIFF,
                        data=graph_diff.model_dump(mode="json"),
                    ).model_dump(mode="json")
                )

            # 3. Send spreading activation waves
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
        # Auto-save conversation on disconnect (if there were messages)
        if len(conversation) >= 2:
            try:
                msgs = [{"role": m.role.value, "content": m.content} for m in conversation]
                save_conversation(user_id, msgs)
            except Exception as exc:
                logger.error("Failed to save conversation: %s", exc)


async def _run_agent(
    user_id: str, messages: list[ChatMessage]
) -> dict:
    """Invoke the LangGraph agent and return the final state."""
    initial_state = {
        "messages": messages,
        "user_id": user_id,
    }
    # LangGraph compiled graphs support async invocation
    result = await _agent.ainvoke(initial_state)
    return result
