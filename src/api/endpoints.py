from fastapi import APIRouter, HTTPException, Request, Header, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime
import uuid
from typing import Optional

from src.core.config import config
from src.core.logging import logger
from src.core.client import OpenAIClient
from src.core.claude_cli_client import ClaudeCliClient
from src.models.claude import ClaudeMessagesRequest, ClaudeTokenCountRequest, ClaudeModelsResponse, ClaudeModelInfo
from src.conversion.request_converter import convert_claude_to_openai
from src.conversion.response_converter import (
    convert_openai_to_claude_response,
    convert_openai_streaming_to_claude_with_cancellation,
)
from src.core.model_manager import model_manager

router = APIRouter()

# Get custom headers from config
custom_headers = config.get_custom_headers()

openai_client = OpenAIClient(
    timeout=config.request_timeout,
    api_version=config.azure_api_version,
    custom_headers=custom_headers,
)

claude_cli_client = ClaudeCliClient(config)


def _build_cli_error_response(e: HTTPException) -> dict:
    """Build an Anthropic-style error response from a CLI HTTPException.

    When the CLI client raises an HTTPException, the detail may be either a
    plain string (for internal errors) or a dict with 'error_type' and
    'message' keys (for mapped Anthropic-style errors like rate_limit_error).
    """
    detail = e.detail
    if isinstance(detail, dict) and "error_type" in detail:
        return {
            "type": "error",
            "error": {
                "type": detail["error_type"],
                "message": detail.get("message", str(detail)),
            },
        }
    # Fallback for non-dict details (string errors)
    error_message = claude_cli_client.classify_cli_error(detail)
    return {
        "type": "error",
        "error": {"type": "api_error", "message": error_message},
    }


async def validate_api_key(x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    """Validate the client's API key from either x-api-key header or Authorization header."""
    client_api_key = None
    
    # Extract API key from headers
    if x_api_key:
        client_api_key = x_api_key
    elif authorization and authorization.startswith("Bearer "):
        client_api_key = authorization.replace("Bearer ", "")
    
    # Skip validation if ANTHROPIC_API_KEY is not set in the environment
    if not config.anthropic_api_key:
        return
        
    # Validate the client API key
    if not client_api_key or not config.validate_client_api_key(client_api_key):
        logger.warning(f"Invalid API key provided by client")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Please provide a valid Anthropic API key."
        )

@router.post("/v1/messages")
async def create_message(request: ClaudeMessagesRequest, http_request: Request, _: None = Depends(validate_api_key)):
    try:
        logger.info(
            f"Received Claude request: model={request.model}, stream={request.stream}"
        )

        # Generate unique request ID for cancellation tracking
        request_id = str(uuid.uuid4())

        # Resolve routing target for the requested model
        target = model_manager.get_model_config(request.model)
        logger.info(
            f"Routing model='{request.model}' -> provider='{target.provider}', backend_model='{target.model}'"
        )

        # Check if client disconnected before processing
        if await http_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        # ------------------------------------------------------------------
        # claude-cli backend
        # ------------------------------------------------------------------
        if target.is_claude_cli:
            if request.stream:
                try:
                    return StreamingResponse(
                        claude_cli_client.create_streaming_completion(
                            request, target, request_id
                        ),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "Access-Control-Allow-Origin": "*",
                            "Access-Control-Allow-Headers": "*",
                        },
                    )
                except HTTPException as e:
                    logger.error(f"Claude CLI streaming error: {e.detail}")
                    error_response = _build_cli_error_response(e)
                    return JSONResponse(
                        status_code=e.status_code, content=error_response
                    )
            else:
                return await claude_cli_client.create_completion(
                    request, target, request_id
                )

        # ------------------------------------------------------------------
        # openai backend (default / original flow)
        # ------------------------------------------------------------------
        openai_request = convert_claude_to_openai(request, model_manager)

        if request.stream:
            # Streaming response - wrap in error handling
            try:
                openai_stream = openai_client.create_chat_completion_stream(
                    openai_request, target.api_key, target.base_url, request_id
                )
                return StreamingResponse(
                    convert_openai_streaming_to_claude_with_cancellation(
                        openai_stream,
                        request,
                        logger,
                        http_request,
                        openai_client,
                        request_id,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*",
                    },
                )
            except HTTPException as e:
                # Convert to proper error response for streaming
                logger.error(f"Streaming error: {e.detail}")
                import traceback

                logger.error(traceback.format_exc())
                error_message = openai_client.classify_openai_error(e.detail)
                error_response = {
                    "type": "error",
                    "error": {"type": "api_error", "message": error_message},
                }
                return JSONResponse(status_code=e.status_code, content=error_response)
        else:
            # Non-streaming response
            openai_response = await openai_client.create_chat_completion(
                openai_request, target.api_key, target.base_url, request_id
            )
            claude_response = convert_openai_to_claude_response(
                openai_response, request
            )
            return claude_response
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        logger.error(f"Unexpected error processing request: {e}")
        logger.error(traceback.format_exc())
        error_message = openai_client.classify_openai_error(str(e))
        raise HTTPException(status_code=500, detail=error_message)


@router.post("/v1/messages/count_tokens")
async def count_tokens(request: ClaudeTokenCountRequest, _: None = Depends(validate_api_key)):
    try:
        # For token counting, we'll use a simple estimation
        # In a real implementation, you might want to use tiktoken or similar

        total_chars = 0

        # Count system message characters
        if request.system:
            if isinstance(request.system, str):
                total_chars += len(request.system)
            elif isinstance(request.system, list):
                for block in request.system:
                    if hasattr(block, "text"):
                        total_chars += len(block.text)

        # Count message characters
        for msg in request.messages:
            if msg.content is None:
                continue
            elif isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "text") and block.text is not None:
                        total_chars += len(block.text)

        # Rough estimation: 4 characters per token
        estimated_tokens = max(1, total_chars // 4)

        return {"input_tokens": estimated_tokens}

    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/models")
async def list_models(_: None = Depends(validate_api_key)):
    """List available Claude models for discovery"""
    models = [
        {
            "id": "claude-3-5-sonnet-20241022",
            "display_name": "Claude 3.5 Sonnet",
            "created": 1729555200,
        },
        {
            "id": "claude-3-5-haiku-20241022",
            "display_name": "Claude 3.5 Haiku",
            "created": 1729555200,
        },
        {
            "id": "claude-3-opus-20240229",
            "display_name": "Claude 3 Opus",
            "created": 1709164800,
        },
        {
            "id": "claude-3-sonnet-20240229",
            "display_name": "Claude 3 Sonnet",
            "created": 1709164800,
        },
        {
            "id": "claude-3-haiku-20240307",
            "display_name": "Claude 3 Haiku",
            "created": 1709769600,
        },
    ]
    return {"data": models}


@router.get("/health")
@router.head("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_api_configured": bool(config.openai_api_key),
        "api_key_valid": config.validate_api_key(),
        "client_api_key_validation": bool(config.anthropic_api_key),
        "providers": {
            "big": config.big_model_provider,
            "middle": config.middle_model_provider,
            "small": config.small_model_provider,
        },
    }


@router.get("/test-connection")
async def test_connection():
    """Test API connectivity to the configured backend."""
    try:
        # Resolve the small-model tier to test connectivity
        target = model_manager.get_model_config("claude-3-5-haiku-20241022")

        if target.is_claude_cli:
            return {
                "status": "success",
                "message": "Claude CLI backend configured",
                "provider": target.provider,
                "model_used": target.model,
                "timestamp": datetime.now().isoformat(),
            }

        test_response = await openai_client.create_chat_completion(
            {
                "model": target.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            },
            target.api_key,
            target.base_url,
        )

        return {
            "status": "success",
            "message": "Successfully connected to OpenAI API",
            "provider": target.provider,
            "model_used": target.model,
            "timestamp": datetime.now().isoformat(),
            "response_id": test_response.get("id", "unknown"),
        }

    except Exception as e:
        logger.error(f"API connectivity test failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "failed",
                "error_type": "API Error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
                "suggestions": [
                    "Check your OPENAI_API_KEY is valid",
                    "Verify your API key has the necessary permissions",
                    "Check if you have reached rate limits",
                ],
            },
        )


@router.get("/")
@router.head("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Claude-to-OpenAI API Proxy v1.0.0",
        "status": "running",
        "config": {
            "openai_base_url": config.openai_base_url,
            "max_tokens_limit": config.max_tokens_limit,
            "api_key_configured": bool(config.openai_api_key),
            "client_api_key_validation": bool(config.anthropic_api_key),
            "big_model": config.big_model,
            "big_model_provider": config.big_model_provider,
            "middle_model": config.middle_model,
            "middle_model_provider": config.middle_model_provider,
            "small_model": config.small_model,
            "small_model_provider": config.small_model_provider,
        },
        "endpoints": {
            "messages": "/v1/messages",
            "models": "/v1/models",
            "count_tokens": "/v1/messages/count_tokens",
            "health": "/health",
            "test_connection": "/test-connection",
        },
    }
