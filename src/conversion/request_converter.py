import json
from typing import Dict, Any, List
from venv import logger
from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest, ClaudeMessage
from src.core.config import config
import logging

logger = logging.getLogger(__name__)


def convert_claude_to_openai(
    claude_request: ClaudeMessagesRequest, model_manager
) -> Dict[str, Any]:
    """Convert Claude API request format to OpenAI format."""

    # Map model
    openai_model = model_manager.map_claude_model_to_openai(claude_request.model)

    # Convert messages
    openai_messages = []

    # Add system message if present
    if claude_request.system:
        system_text = ""
        if isinstance(claude_request.system, str):
            system_text = claude_request.system
        elif isinstance(claude_request.system, list):
            text_parts = []
            for block in claude_request.system:
                if hasattr(block, "type") and getattr(block, "type") == Constants.CONTENT_TEXT:
                    text_parts.append(getattr(block, "text", ""))
                elif (
                    isinstance(block, dict)
                    and block.get("type") == Constants.CONTENT_TEXT
                ):
                    text_parts.append(block.get("text", ""))
            system_text = "\n\n".join(text_parts)

        if system_text.strip():
            openai_messages.append(
                {"role": Constants.ROLE_SYSTEM, "content": system_text.strip()}
            )

    # Process Claude messages
    i = 0
    while i < len(claude_request.messages):
        msg = claude_request.messages[i]

        if msg.role == Constants.ROLE_USER:
            openai_message = convert_claude_user_message(msg, openai_model)
            openai_messages.append(openai_message)
        elif msg.role == Constants.ROLE_SYSTEM:
            openai_message = convert_claude_system_message(msg)
            openai_messages.append(openai_message)
        elif msg.role == Constants.ROLE_ASSISTANT:
            openai_message = convert_claude_assistant_message(msg)
            openai_messages.append(openai_message)

            # Check if next message contains tool results
            if i + 1 < len(claude_request.messages):
                next_msg = claude_request.messages[i + 1]
                if (
                    next_msg.role == Constants.ROLE_USER
                    and isinstance(next_msg.content, list)
                    and any(
                        block.type == Constants.CONTENT_TOOL_RESULT
                        for block in next_msg.content
                        if hasattr(block, "type")
                    )
                ):
                    # Process tool results
                    i += 1  # Skip to tool result message
                    tool_results = convert_claude_tool_results(next_msg)
                    openai_messages.extend(tool_results)

        i += 1

    # Build OpenAI request
    openai_request = {
        "model": openai_model,
        "messages": openai_messages,
        "max_tokens": min(
            max(claude_request.max_tokens, config.min_tokens_limit),
            config.max_tokens_limit,
        ),
        "temperature": claude_request.temperature,
        "stream": claude_request.stream,
    }
    logger.debug(
        f"Converted Claude request to OpenAI format: {json.dumps(openai_request, indent=2, ensure_ascii=False)}"
    )
    # Add optional parameters
    if claude_request.stop_sequences:
        openai_request["stop"] = claude_request.stop_sequences
    if claude_request.top_p is not None:
        openai_request["top_p"] = claude_request.top_p

    # Convert tools
    if claude_request.tools:
        openai_tools = []
        for tool in claude_request.tools:
            if tool.name and tool.name.strip():
                openai_tools.append(
                    {
                        "type": Constants.TOOL_FUNCTION,
                        Constants.TOOL_FUNCTION: {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.input_schema,
                        },
                    }
                )
        if openai_tools:
            openai_request["tools"] = openai_tools

    # Convert tool choice
    if claude_request.tool_choice:
        choice_type = claude_request.tool_choice.get("type")
        if choice_type == "auto":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "any":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "tool" and "name" in claude_request.tool_choice:
            openai_request["tool_choice"] = {
                "type": Constants.TOOL_FUNCTION,
                Constants.TOOL_FUNCTION: {"name": claude_request.tool_choice["name"]},
            }
        else:
            openai_request["tool_choice"] = "auto"

    with open("last_openai_request.json", "w") as f:
        json.dump(openai_request, f, indent=2)

    return openai_request


def convert_claude_user_message(msg: ClaudeMessage, target_model: str = "") -> Dict[str, Any]:
    """Convert Claude user message to OpenAI format."""
    if msg.content is None:
        return {"role": Constants.ROLE_USER, "content": ""}
    
    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_USER, "content": msg.content}

    # Handle multimodal content
    openai_content: List[Dict[str, Any]] = []
    for block in msg.content:
        b_type = getattr(block, "type", None)
        if b_type == Constants.CONTENT_TEXT:
            openai_content.append({"type": "text", "text": getattr(block, "text", "")})
        elif b_type == Constants.CONTENT_THINKING:
            openai_content.append({"type": "text", "text": f"<thinking>\n{getattr(block, 'thinking', '')}\n</thinking>"})
        elif b_type == Constants.CONTENT_REDACTED_THINKING:
            openai_content.append({"type": "text", "text": f"<thinking>\n[Redacted Thinking: {getattr(block, 'data', '')}]\n</thinking>"})
        elif b_type == Constants.CONTENT_IMAGE:
            from src.core.config import config
            strip_for_model = any(m in target_model.lower() for m in config.strip_image_models) if target_model else False
            
            if config.strip_images or strip_for_model:
                continue
            
            # Convert Claude image format to OpenAI format
            source = getattr(block, "source", None)
            if source:
                s_type = source.get("type") if isinstance(source, dict) else getattr(source, "type", "")
                s_media = source.get("media_type") if isinstance(source, dict) else getattr(source, "media_type", "")
                s_data = source.get("data") if isinstance(source, dict) else getattr(source, "data", "")
                
                if s_type == "base64" and s_media and s_data:
                    openai_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{s_media};base64,{s_data}"
                            },
                        }
                    )

    has_image_url = any(b.get("type") == "image_url" for b in openai_content)
    if not has_image_url:
        text_content = "\n\n".join(b["text"] for b in openai_content if b.get("type") == "text")
        return {"role": Constants.ROLE_USER, "content": text_content}
    else:
        return {"role": Constants.ROLE_USER, "content": openai_content}


def convert_claude_system_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude system message to OpenAI format."""
    if msg.content is None:
        return {"role": Constants.ROLE_SYSTEM, "content": ""}

    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_SYSTEM, "content": msg.content}

    # Handle list content (similar to user message but for system)
    text_parts = []
    for block in msg.content:
        if getattr(block, "type", None) == Constants.CONTENT_TEXT:
            text_parts.append(getattr(block, "text", ""))

    return {"role": Constants.ROLE_SYSTEM, "content": "\n\n".join(text_parts)}


def convert_claude_assistant_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude assistant message to OpenAI format."""
    text_parts = []
    tool_calls = []

    if msg.content is None:
        return {"role": Constants.ROLE_ASSISTANT, "content": None}
    
    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_ASSISTANT, "content": msg.content}

    for block in msg.content:
        b_type = getattr(block, "type", None)
        if b_type == Constants.CONTENT_TEXT:
            text_parts.append(getattr(block, "text", ""))
        elif b_type == Constants.CONTENT_THINKING:
            text_parts.append(f"<thinking>\n{getattr(block, 'thinking', '')}\n</thinking>")
        elif b_type == Constants.CONTENT_REDACTED_THINKING:
            text_parts.append(f"<thinking>\n[Redacted Thinking: {getattr(block, 'data', '')}]\n</thinking>")
        elif b_type == Constants.CONTENT_TOOL_USE:
            tool_calls.append(
                {
                    "id": getattr(block, "id", ""),
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": getattr(block, "name", ""),
                        "arguments": json.dumps(getattr(block, "input", {}), ensure_ascii=False),
                    },
                }
            )

    openai_message: Dict[str, Any] = {"role": Constants.ROLE_ASSISTANT}

    # Set content
    if text_parts:
        openai_message["content"] = "".join(text_parts)
    else:
        openai_message["content"] = ""

    # Set tool calls
    if tool_calls:
        openai_message["tool_calls"] = tool_calls

    return openai_message


def convert_claude_tool_results(msg: ClaudeMessage) -> List[Dict[str, Any]]:
    """Convert Claude tool results to OpenAI format."""
    tool_messages = []

    if isinstance(msg.content, list):
        for block in msg.content:
            if getattr(block, "type", None) == Constants.CONTENT_TOOL_RESULT:
                content = parse_tool_result_content(getattr(block, "content", ""))
                tool_messages.append(
                    {
                        "role": Constants.ROLE_TOOL,
                        "tool_call_id": getattr(block, "tool_use_id", ""),
                        "content": content,
                    }
                )

    return tool_messages


def parse_tool_result_content(content):
    """Parse and normalize tool result content into a string format."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == Constants.CONTENT_TEXT:
                result_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                result_parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    result_parts.append(item.get("text", ""))
                else:
                    try:
                        result_parts.append(json.dumps(item, ensure_ascii=False))
                    except:
                        result_parts.append(str(item))
        return "\n".join(result_parts).strip()

    if isinstance(content, dict):
        if content.get("type") == Constants.CONTENT_TEXT:
            return content.get("text", "")
        try:
            return json.dumps(content, ensure_ascii=False)
        except:
            return str(content)

    try:
        return str(content)
    except:
        return "Unparseable content"
