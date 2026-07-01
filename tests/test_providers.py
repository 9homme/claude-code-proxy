"""Tests for provider routing (openai vs claude-cli) and CLI result conversion."""

import os
import pytest
from unittest.mock import patch, MagicMock

# Ensure required env vars exist before importing Config
os.environ.setdefault("OPENAI_API_KEY", "sk-test")


def _make_config(overrides: dict, clear_providers: bool = False) -> "Config":
    """Build a Config instance with the given env overrides applied.

    When clear_providers is True, provider env vars are removed first so the
    real .env doesn't leak into default-provider tests.
    """
    from src.core.config import Config

    env = dict(os.environ)
    if clear_providers:
        for var in (
            "BIG_MODEL_PROVIDER",
            "MIDDLE_MODEL_PROVIDER",
            "SMALL_MODEL_PROVIDER",
        ):
            env.pop(var, None)
    env.update(overrides)

    with patch.dict(os.environ, env, clear=True):
        return Config()


def test_default_providers_are_openai():
    """Without provider env vars, all tiers default to the openai provider."""
    from src.core.model_target import PROVIDER_OPENAI

    cfg = _make_config({}, clear_providers=True)
    from src.core.model_manager import ModelManager

    manager = ModelManager(cfg)
    for model in ["claude-3-opus-20240229", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]:
        target = manager.get_model_config(model)
        assert target.provider == PROVIDER_OPENAI, f"{model} should default to openai"


def test_claude_cli_provider_selected_per_tier():
    """Setting *_MODEL_PROVIDER=claude-cli routes that tier to the CLI."""
    from src.core.model_target import PROVIDER_CLAUDE_CLI

    cfg = _make_config(
        {
            "BIG_MODEL_PROVIDER": "claude-cli",
            "BIG_MODEL": "opus",
            "MIDDLE_MODEL_PROVIDER": "claude-cli",
            "MIDDLE_MODEL": "sonnet",
            "SMALL_MODEL_PROVIDER": "claude-cli",
            "SMALL_MODEL": "haiku",
        }
    )
    from src.core.model_manager import ModelManager

    manager = ModelManager(cfg)

    opus_target = manager.get_model_config("claude-3-opus-20240229")
    assert opus_target.is_claude_cli
    assert opus_target.model == "opus"

    sonnet_target = manager.get_model_config("claude-3-5-sonnet-20241022")
    assert sonnet_target.is_claude_cli
    assert sonnet_target.model == "sonnet"

    haiku_target = manager.get_model_config("claude-3-5-haiku-20241022")
    assert haiku_target.is_claude_cli
    assert haiku_target.model == "haiku"


def test_mixed_providers():
    """Mixing providers across tiers resolves correctly."""
    from src.core.model_target import PROVIDER_OPENAI, PROVIDER_CLAUDE_CLI

    cfg = _make_config(
        {
            "BIG_MODEL_PROVIDER": "claude-cli",
            "BIG_MODEL": "opus",
            "MIDDLE_MODEL_PROVIDER": "openai",
            "MIDDLE_MODEL": "gpt-4o",
            "SMALL_MODEL_PROVIDER": "openai",
            "SMALL_MODEL": "gpt-4o-mini",
        }
    )
    from src.core.model_manager import ModelManager

    manager = ModelManager(cfg)

    big = manager.get_model_config("claude-3-opus-20240229")
    assert big.provider == PROVIDER_CLAUDE_CLI
    assert big.is_claude_cli

    middle = manager.get_model_config("claude-3-5-sonnet-20241022")
    assert middle.provider == PROVIDER_OPENAI
    assert middle.is_openai

    small = manager.get_model_config("claude-3-5-haiku-20241022")
    assert small.provider == PROVIDER_OPENAI


def test_map_claude_model_to_openai_returns_model_name():
    """The compatibility helper should still return a model name string."""
    cfg = _make_config({})
    from src.core.model_manager import ModelManager

    manager = ModelManager(cfg)
    assert isinstance(manager.map_claude_model_to_openai("claude-3-opus-20240229"), str)


def test_cli_result_to_claude_response_success():
    """A successful CLI result is converted to a Claude Messages response."""
    from src.core.claude_cli_client import ClaudeCliClient
    from src.models.claude import ClaudeMessagesRequest

    cfg = _make_config({})
    client = ClaudeCliClient(cfg)

    cli_result = {
        "type": "result",
        "is_error": False,
        "result": "Hello there!",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }
    request = ClaudeMessagesRequest(model="claude-3-opus-20240229", max_tokens=100, messages=[])

    response = client._result_to_claude_response(cli_result, request)
    assert response["type"] == "message"
    assert response["role"] == "assistant"
    assert response["content"] == [{"type": "text", "text": "Hello there!"}]
    assert response["usage"] == {"input_tokens": 10, "output_tokens": 3}
    assert response["stop_reason"] == "end_turn"


def test_cli_result_to_claude_response_error_raises():
    """A CLI-level error result should raise an HTTPException with type info."""
    from fastapi import HTTPException
    from src.core.claude_cli_client import ClaudeCliClient
    from src.models.claude import ClaudeMessagesRequest

    cfg = _make_config({})
    client = ClaudeCliClient(cfg)

    cli_result = {
        "type": "result",
        "is_error": True,
        "api_error_status": 429,
        "result": "You've hit your session limit",
    }
    request = ClaudeMessagesRequest(model="claude-3-opus-20240229", max_tokens=100, messages=[])

    with pytest.raises(HTTPException) as exc_info:
        client._result_to_claude_response(cli_result, request)
    assert exc_info.value.status_code == 429
    # The detail should be a dict with Anthropic-style error info
    detail = exc_info.value.detail
    assert isinstance(detail, dict)
    assert detail["error_type"] == "rate_limit_error"
    assert "session limit" in detail["message"]


def test_classify_cli_error_types():
    """_classify_cli_error maps common CLI errors to Anthropic error types."""
    from src.core.claude_cli_client import ClaudeCliClient

    cfg = _make_config({})
    client = ClaudeCliClient(cfg)

    # Rate limit / session limit
    err = client._classify_cli_error(
        {"is_error": True, "api_error_status": 429, "result": "session limit reached"}
    )
    assert err["error_type"] == "rate_limit_error"
    assert err["status_code"] == 429

    # Auth error
    err = client._classify_cli_error(
        {"is_error": True, "api_error_status": 401, "result": "Invalid API key"}
    )
    assert err["error_type"] == "authentication_error"
    assert err["status_code"] == 401

    # Bad request
    err = client._classify_cli_error(
        {"is_error": True, "api_error_status": 400, "result": "Bad input"}
    )
    assert err["error_type"] == "invalid_request_error"
    assert err["status_code"] == 400

    # Overloaded
    err = client._classify_cli_error(
        {"is_error": True, "api_error_status": 503, "result": "Overloaded"}
    )
    assert err["error_type"] == "overloaded_error"
    assert err["status_code"] == 503


def test_sse_error_typed():
    """_sse_error_typed should produce Anthropic-style error SSE events."""
    import json
    from src.core.claude_cli_client import ClaudeCliClient

    sse = ClaudeCliClient._sse_error_typed("rate_limit_error", "session limit reached")
    assert sse.startswith("event: error\n")
    # Parse the data line
    data_line = sse.split("\n")[1].replace("data: ", "")
    parsed = json.loads(data_line)
    assert parsed["type"] == "error"
    assert parsed["error"]["type"] == "rate_limit_error"
    assert parsed["error"]["message"] == "session limit reached"


def test_build_command_includes_model_and_stream_flags():
    """The constructed CLI command should include model + format flags."""
    from src.core.claude_cli_client import ClaudeCliClient

    cfg = _make_config({})
    client = ClaudeCliClient(cfg)

    stream_cmd = client._build_command("sonnet", stream=True)
    assert "--model" in stream_cmd
    assert "sonnet" in stream_cmd
    assert "--output-format" in stream_cmd
    assert "stream-json" in stream_cmd

    non_stream_cmd = client._build_command("opus", stream=False)
    assert "json" in non_stream_cmd
    assert "stream-json" not in non_stream_cmd


def test_build_command_no_session_persistence():
    """Command should use --no-session-persistence."""
    from src.core.claude_cli_client import ClaudeCliClient

    cfg = _make_config({})
    client = ClaudeCliClient(cfg)

    cmd = client._build_command("opus", stream=False)
    assert "--no-session-persistence" in cmd
    # Must not use --system-prompt or --append-system-prompt
    assert "--system-prompt" not in cmd
    assert "--append-system-prompt" not in cmd


def test_build_command_with_session_id():
    """Command should include --session-id when provided."""
    from src.core.claude_cli_client import ClaudeCliClient

    cfg = _make_config({})
    client = ClaudeCliClient(cfg)

    cmd = client._build_command("opus", stream=False, session_id="test-uuid-123")
    assert "--session-id" in cmd
    idx = cmd.index("--session-id")
    assert cmd[idx + 1] == "test-uuid-123"


def test_build_command_tools_disabled_by_default():
    """By default, --allowedTools '' should disable tools (pure inference)."""
    from src.core.claude_cli_client import ClaudeCliClient

    # Explicitly clear CLAUDE_CLI_ENABLE_TOOLS so the real .env doesn't leak
    cfg = _make_config({"CLAUDE_CLI_ENABLE_TOOLS": "false"})
    client = ClaudeCliClient(cfg)

    cmd = client._build_command("opus", stream=False)
    assert "--allowedTools" in cmd
    idx = cmd.index("--allowedTools")
    assert cmd[idx + 1] == ""


def test_build_command_tools_enabled():
    """When CLAUDE_CLI_ENABLE_TOOLS=true, tools kept and --append-system-prompt used."""
    from src.core.claude_cli_client import ClaudeCliClient

    cfg = _make_config({"CLAUDE_CLI_ENABLE_TOOLS": "true"})
    client = ClaudeCliClient(cfg)

    # Without system prompt
    cmd = client._build_command("opus", stream=False)
    assert "--allowedTools" not in cmd, "Tools should be enabled (no --allowedTools flag)"
    assert "--append-system-prompt" not in cmd  # No system prompt provided

    # With system prompt → should use --append-system-prompt
    cmd = client._build_command("opus", stream=False, system_prompt="Be helpful")
    assert "--allowedTools" not in cmd
    assert "--append-system-prompt" in cmd
    idx = cmd.index("--append-system-prompt")
    assert cmd[idx + 1] == "Be helpful"


def test_conversation_prompt_embeds_system_prompt():
    """System prompt should be embedded in the stdin text."""
    from src.core.claude_cli_client import ClaudeCliClient
    from src.models.claude import ClaudeMessage, ClaudeMessagesRequest

    cfg = _make_config({})
    client = ClaudeCliClient(cfg)

    request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        system="You are a helpful assistant.",
        messages=[ClaudeMessage(role="user", content="Hello!")],
    )
    system_prompt = client._build_system_prompt(request)
    prompt = client._build_conversation_prompt(request, system_prompt=system_prompt)

    # System prompt SHOULD be in the conversation text
    assert "You are a helpful assistant." in prompt
    assert "Hello!" in prompt


def test_classify_cli_error_messages():
    """classify_cli_error should return friendly messages for known failures."""
    from src.core.claude_cli_client import ClaudeCliClient

    cfg = _make_config({})
    client = ClaudeCliClient(cfg)

    assert "rate" in client.classify_cli_error("You've hit your session limit").lower()
    assert "authenticated" in client.classify_cli_error("not logged in").lower()
    assert "not found" in client.classify_cli_error("command not found").lower()
    assert "auth conflict" in client.classify_cli_error("Invalid API key · Fix external API key").lower()


def test_cli_env_strips_anthropic_vars():
    """_get_cli_env should strip ANTHROPIC_API_KEY/BASE_URL/CLAUDECODE for CLI."""
    from src.core.claude_cli_client import ClaudeCliClient

    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_API_KEY": "sk-fake",
            "ANTHROPIC_BASE_URL": "http://localhost:8082",
            "CLAUDECODE": "1",
        },
    ):
        cfg = _make_config({})
        client = ClaudeCliClient(cfg)
        env = client._get_cli_env()

    assert "ANTHROPIC_API_KEY" not in env, "ANTHROPIC_API_KEY must be stripped for CLI"
    assert "ANTHROPIC_BASE_URL" not in env, "ANTHROPIC_BASE_URL must be stripped for CLI"
    assert "ANTHROPIC_AUTH_TOKEN" not in env, "ANTHROPIC_AUTH_TOKEN must be stripped for CLI"
    assert "CLAUDECODE" not in env, "CLAUDECODE must be stripped for CLI"


def test_session_manager_single_message_returns_none():
    """Single-message conversations should not get a session ID."""
    from src.core.session_manager import SessionManager
    from src.models.claude import ClaudeMessage

    sm = SessionManager()
    msgs = [ClaudeMessage(role="user", content="Hello")]
    assert sm.get_or_create(msgs, "opus") is None


def test_session_manager_multi_turn_creates_and_reuses():
    """Multi-turn conversations should get a session ID, reused on follow-up."""
    from src.core.session_manager import SessionManager
    from src.models.claude import ClaudeMessage

    sm = SessionManager()
    msgs1 = [
        ClaudeMessage(role="user", content="Hello"),
        ClaudeMessage(role="assistant", content="Hi there!"),
        ClaudeMessage(role="user", content="How are you?"),
    ]
    session1 = sm.get_or_create(msgs1, "opus")
    assert session1 is not None
    assert len(session1) > 0

    # Same conversation prefix → same session
    msgs2 = [
        ClaudeMessage(role="user", content="Hello"),
        ClaudeMessage(role="assistant", content="Hi there!"),
        ClaudeMessage(role="user", content="What's the weather?"),
    ]
    session2 = sm.get_or_create(msgs2, "opus")
    assert session2 == session1, "Same conversation prefix should reuse session"
