"""Claude CLI backend client.

Runs the `claude` CLI (Claude Code) in print mode to serve inference requests
using a Claude Code Max subscription. Supports streaming and non-streaming
responses, and produces output in Claude Messages API format.
"""

import asyncio
import json
import os
import time
import uuid
from typing import AsyncGenerator, Dict, Any, Optional, List

from fastapi import HTTPException

from src.core.config import Config
from src.core.constants import Constants
from src.core.logging import logger
from src.core.model_target import ModelTarget
from src.core.session_manager import session_manager
from src.models.claude import ClaudeMessagesRequest


class ClaudeCliClient:
    """Async client that invokes the `claude` CLI for inference."""

    def __init__(self, config: Config):
        self.config = config
        self.active_processes: Dict[str, asyncio.subprocess.Process] = {}

    def _get_cli_env(self) -> Dict[str, str]:
        """Build a sanitized environment for the claude CLI subprocess.

        The proxy sets ANTHROPIC_API_KEY (for client validation) and
        ANTHROPIC_BASE_URL (to redirect clients to this proxy). If these leak
        into the CLI subprocess, the CLI tries to use them for authentication
        instead of the OAuth subscription login. We strip them out so the CLI
        falls back to the keychain/OAuth credentials.

        We also strip CLAUDECODE, which the CLI uses to detect when it's
        running inside another Claude Code instance.
        """
        env = dict(os.environ)
        # Remove proxy/auth env vars that interfere with the CLI's own auth
        for key in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_AUTH_TOKEN",
            "CLAUDECODE",
        ):
            env.pop(key, None)
        return env

    # ------------------------------------------------------------------
    # Input building helpers
    # ------------------------------------------------------------------

    def _extract_system_prompt(self, request: ClaudeMessagesRequest) -> str:
        """Extract and combine system prompt text from a Claude request."""
        if not request.system:
            return ""
        if isinstance(request.system, str):
            return request.system.strip()
        # List of content blocks
        parts: List[str] = []
        for block in request.system:
            if hasattr(block, "text"):
                parts.append(getattr(block, "text", ""))
            elif isinstance(block, dict) and block.get("type") == Constants.CONTENT_TEXT:
                parts.append(block.get("text", ""))
        return "\n\n".join(parts).strip()

    def _extract_message_text(self, msg) -> str:
        """Extract a plain-text representation from a Claude message."""
        if msg.content is None:
            return ""
        if isinstance(msg.content, str):
            return msg.content

        text_parts: List[str] = []
        for block in msg.content:
            b_type = getattr(block, "type", None)
            if b_type == Constants.CONTENT_TEXT:
                text_parts.append(getattr(block, "text", ""))
            elif b_type == Constants.CONTENT_TOOL_USE:
                name = getattr(block, "name", "")
                inp = getattr(block, "input", {})
                text_parts.append(f"[Tool call: {name}({json.dumps(inp, ensure_ascii=False)})]")
            elif b_type == Constants.CONTENT_TOOL_RESULT:
                content = getattr(block, "content", "")
                if isinstance(content, str):
                    text_parts.append(f"[Tool result: {content}]")
                else:
                    text_parts.append(f"[Tool result: {json.dumps(content, ensure_ascii=False)}]")
            elif b_type == Constants.CONTENT_THINKING:
                thinking = getattr(block, "thinking", "")
                text_parts.append(f"[Thinking: {thinking}]")
            elif b_type == Constants.CONTENT_REDACTED_THINKING:
                data = getattr(block, "data", "")
                text_parts.append(f"[Redacted thinking: {data}]")
        return "\n".join(text_parts)

    def _build_tools_description(self, request: ClaudeMessagesRequest) -> Optional[str]:
        """Convert tool definitions to a text description for the system prompt."""
        if not request.tools:
            return None

        parts = ["\n\n--- Tools available ---\nYou have access to the following tools:"]
        for tool in request.tools:
            if not tool.name or not tool.name.strip():
                continue
            desc = f"\n  - {tool.name}: {tool.description or ''}"
            desc += f"\n    Parameters: {json.dumps(tool.input_schema, ensure_ascii=False)}"
            parts.append(desc)
        parts.append(
            "\nWhen you want to use a tool, respond with a JSON block in this format:\n"
            '```json\n{"tool": "<tool_name>", "input": {<parameters>}}\n```'
        )
        return "\n".join(parts)

    def _build_system_prompt(self, request: ClaudeMessagesRequest) -> str:
        """Build the complete system prompt including tool descriptions."""
        base = self._extract_system_prompt(request) or ""
        tools_desc = self._build_tools_description(request)
        if tools_desc:
            return base + tools_desc
        return base

    def _build_conversation_prompt(
        self, request: ClaudeMessagesRequest
    ) -> str:
        """Flatten Claude messages into a text prompt for the CLI.

        The system prompt is passed separately via --append-system-prompt (see
        _build_command), so it is NOT embedded here. Only the user/assistant
        message content is included in the stdin text.
        """
        messages = request.messages
        if not messages:
            return ""

        # Single user message — pass through directly
        if len(messages) == 1 and messages[0].role == Constants.ROLE_USER:
            return self._extract_message_text(messages[0])

        # Multi-turn: build a labelled transcript
        parts: List[str] = []
        for msg in messages:
            role = msg.role.upper()
            text = self._extract_message_text(msg)
            if text:
                parts.append(f"[{role}]\n{text}")

        transcript = "\n\n".join(parts)
        return (
            f"<conversation>\n{transcript}\n</conversation>\n\n"
            "Please respond to the last message in the conversation above."
        )

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def _build_command(
        self, model: str, stream: bool, system_prompt: str = "",
        session_id: Optional[str] = None,
    ) -> List[str]:
        """Build the claude CLI command.

        The system prompt is passed via --append-system-prompt, which appends
        to Claude Code's default system prompt. This keeps the model's built-in
        context while adding our instructions. For very large system prompts,
        --append-system-prompt may still hit OS arg limits; in that case,
        callers should embed the system prompt in stdin text instead.

        Session persistence:
        - --no-session-persistence prevents the CLI from saving sessions to disk
        - --session-id passes a CLI session UUID for multi-turn context reuse
        """
        cmd: List[str] = [
            self.config.claude_cli_path,
            "-p",
            "--model", model,
            # Don't save sessions to disk — the proxy manages session IDs
            "--no-session-persistence",
        ]

        if stream:
            cmd.extend([
                "--output-format", "stream-json",
                "--include-partial-messages",
                "--verbose",
            ])
        else:
            cmd.extend(["--output-format", "json"])

        # Pass system prompt via --append-system-prompt (keeps Claude Code's
        # default system prompt and appends ours)
        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])

        # Pass session ID for multi-turn context reuse
        if session_id:
            cmd.extend(["--session-id", session_id])

        # Disable all built-in tools — we want pure text inference.
        cmd.extend(["--allowedTools", ""])

        if self.config.claude_cli_skip_permissions:
            cmd.append("--dangerously-skip-permissions")

        cmd.extend(self.config.claude_cli_extra_args)

        return cmd

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def create_completion(
        self,
        request: ClaudeMessagesRequest,
        target: ModelTarget,
        request_id: Optional[str] = None,
    ) -> dict:
        """Run the CLI in non-streaming mode and return a Claude-format dict."""
        system_prompt = self._build_system_prompt(request)
        prompt = self._build_conversation_prompt(request)
        session_id = session_manager.get_or_create(request.messages, target.model)
        cmd = self._build_command(
            target.model, stream=False,
            system_prompt=system_prompt, session_id=session_id,
        )

        start_time = time.time()
        logger.info(f"Claude CLI request started (model: {target.model})")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._get_cli_env(),
            )
            if request_id:
                self.active_processes[request_id] = proc

            stdout_data, stderr_data = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=self.config.claude_cli_timeout,
            )

            duration = time.time() - start_time
            logger.info(f"Claude CLI request completed in {duration:.2f}s (model: {target.model})")

            stdout_text = stdout_data.decode("utf-8", errors="replace").strip()
            stderr_text = stderr_data.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0:
                logger.error(f"Claude CLI exit code {proc.returncode}")
                logger.error(f"Claude CLI stderr: {stderr_text[:1000]}")
                logger.error(f"Claude CLI stdout: {stdout_text[:1000]}")
                raise HTTPException(
                    status_code=502,
                    detail=f"Claude CLI failed (exit {proc.returncode}): {stderr_text[:500] or stdout_text[:500]}",
                )

            # Parse the JSON result line
            try:
                result = json.loads(stdout_text)
            except json.JSONDecodeError:
                logger.error(f"Claude CLI returned non-JSON output: {stdout_text[:1000]}")
                raise HTTPException(
                    status_code=502,
                    detail=f"Claude CLI returned invalid JSON: {stdout_text[:500]}",
                )

            # Log the raw result for diagnostics before converting
            if result.get("is_error"):
                logger.warning(f"Claude CLI result signaled error: {json.dumps(result, ensure_ascii=False)[:1000]}")

            return self._result_to_claude_response(result, request)

        except asyncio.TimeoutError:
            logger.error(f"Claude CLI timeout after {self.config.claude_cli_timeout}s")
            raise HTTPException(
                status_code=504,
                detail=f"Claude CLI timed out after {self.config.claude_cli_timeout}s",
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Claude CLI error: {e}")
            raise HTTPException(status_code=500, detail=f"Claude CLI error: {str(e)}")
        finally:
            if request_id and request_id in self.active_processes:
                del self.active_processes[request_id]

    def _classify_cli_error(self, cli_result: dict) -> Dict[str, Any]:
        """Map a CLI error result to an Anthropic-style error descriptor.

        Returns a dict with keys: error_type, message, status_code.
        Anthropic error types: rate_limit_error, authentication_error,
        invalid_request_error, overloaded_error, api_error.
        """
        error_msg = str(cli_result.get("result", "Unknown CLI error"))
        error_lower = error_msg.lower()
        api_status = cli_result.get("api_error_status")
        status_code = api_status if isinstance(api_status, int) else 502

        # Map common CLI errors to Anthropic error types
        if "session limit" in error_lower or status_code == 429:
            return {
                "error_type": "rate_limit_error",
                "message": error_msg,
                "status_code": 429,
            }
        if "invalid api key" in error_lower or "fix external api key" in error_lower:
            return {
                "error_type": "authentication_error",
                "message": error_msg,
                "status_code": 401,
            }
        if status_code == 401:
            return {
                "error_type": "authentication_error",
                "message": error_msg,
                "status_code": 401,
            }
        if status_code == 400:
            return {
                "error_type": "invalid_request_error",
                "message": error_msg,
                "status_code": 400,
            }
        if status_code in (500, 502, 503):
            return {
                "error_type": "overloaded_error",
                "message": error_msg,
                "status_code": status_code,
            }
        # Default
        return {
            "error_type": "api_error",
            "message": error_msg,
            "status_code": status_code,
        }

    def _raise_cli_error(self, cli_result: dict):
        """Raise an HTTPException with Anthropic-style error info in detail."""
        err = self._classify_cli_error(cli_result)
        raise HTTPException(
            status_code=err["status_code"],
            detail=err,  # Pass full error info as dict
        )

    def _result_to_claude_response(
        self, cli_result: dict, original_request: ClaudeMessagesRequest
    ) -> dict:
        """Convert a claude CLI JSON result into a Claude Messages API dict."""
        # Check for CLI-level errors (e.g. rate limits)
        if cli_result.get("is_error"):
            self._raise_cli_error(cli_result)

        text = cli_result.get("result", "") or ""
        usage = cli_result.get("usage", {})
        stop_reason_raw = cli_result.get("stop_reason", "")

        # Map stop reasons
        stop_reason_map = {
            "end_turn": Constants.STOP_END_TURN,
            "stop_sequence": Constants.STOP_END_TURN,
            "max_tokens": Constants.STOP_MAX_TOKENS,
            "tool_use": Constants.STOP_TOOL_USE,
        }
        stop_reason = stop_reason_map.get(stop_reason_raw, Constants.STOP_END_TURN)

        return {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": Constants.ROLE_ASSISTANT,
            "model": original_request.model,
            "content": [{"type": Constants.CONTENT_TEXT, "text": text}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
        }

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def create_streaming_completion(
        self,
        request: ClaudeMessagesRequest,
        target: ModelTarget,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Run the CLI in streaming mode and yield Claude SSE events."""
        system_prompt = self._build_system_prompt(request)
        prompt = self._build_conversation_prompt(request)
        session_id = session_manager.get_or_create(request.messages, target.model)
        cmd = self._build_command(
            target.model, stream=True,
            system_prompt=system_prompt, session_id=session_id,
        )

        start_time = time.time()
        logger.info(f"Claude CLI stream started (model: {target.model})")

        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        ttft_logged = False
        usage_data: Dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}

        # Emit initial SSE events
        yield self._sse(
            Constants.EVENT_MESSAGE_START,
            {
                "type": Constants.EVENT_MESSAGE_START,
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": Constants.ROLE_ASSISTANT,
                    "model": request.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )
        yield self._sse(
            Constants.EVENT_CONTENT_BLOCK_START,
            {
                "type": Constants.EVENT_CONTENT_BLOCK_START,
                "index": 0,
                "content_block": {"type": Constants.CONTENT_TEXT, "text": ""},
            },
        )
        yield self._sse(Constants.EVENT_PING, {"type": Constants.EVENT_PING})

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._get_cli_env(),
            )
            if request_id:
                self.active_processes[request_id] = proc

            # Write prompt to stdin and close it
            proc.stdin.write(prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

            text_block_index = 0
            final_stop_reason = Constants.STOP_END_TURN

            assert proc.stdout is not None
            async for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Unparseable CLI stream line: {line[:200]}")
                    continue

                if not ttft_logged:
                    ttft = time.time() - start_time
                    logger.info(
                        f"Claude CLI stream first event (TTFT: {ttft:.2f}s, model: {target.model})"
                    )
                    ttft_logged = True

                event_type = event.get("type")

                # Handle stream_event wrapper (partial messages from the API)
                if event_type == "stream_event":
                    inner = event.get("event", {})
                    inner_type = inner.get("type")

                    if inner_type == "content_block_delta":
                        delta = inner.get("delta", {})
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            yield self._sse(
                                Constants.EVENT_CONTENT_BLOCK_DELTA,
                                {
                                    "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                                    "index": text_block_index,
                                    "delta": {
                                        "type": Constants.DELTA_TEXT,
                                        "text": delta["text"],
                                    },
                                },
                            )

                    elif inner_type == "message_delta":
                        delta = inner.get("delta", {})
                        sr = delta.get("stop_reason")
                        if sr == "max_tokens":
                            final_stop_reason = Constants.STOP_MAX_TOKENS
                        elif sr == "tool_use":
                            final_stop_reason = Constants.STOP_TOOL_USE
                        inner_usage = inner.get("usage", {})
                        if inner_usage:
                            usage_data["input_tokens"] = inner_usage.get(
                                "input_tokens", usage_data.get("input_tokens", 0)
                            )
                            usage_data["output_tokens"] = inner_usage.get(
                                "output_tokens", usage_data.get("output_tokens", 0)
                            )

                # Handle the final result event
                elif event_type == "result":
                    if event.get("is_error"):
                        err = self._classify_cli_error(event)
                        yield self._sse_error_typed(
                            err["error_type"], err["message"]
                        )
                        return

                    r_usage = event.get("usage", {})
                    if r_usage:
                        usage_data["input_tokens"] = r_usage.get(
                            "input_tokens", usage_data.get("input_tokens", 0)
                        )
                        usage_data["output_tokens"] = r_usage.get(
                            "output_tokens", usage_data.get("output_tokens", 0)
                        )

                    sr = event.get("stop_reason", "")
                    if sr == "max_tokens":
                        final_stop_reason = Constants.STOP_MAX_TOKENS
                    elif sr == "tool_use":
                        final_stop_reason = Constants.STOP_TOOL_USE

                    break  # result is always the last event

                # 'assistant' and 'user' (tool result) events from the CLI can
                # be safely ignored since we disabled built-in tools and we
                # already streamed partial deltas above.
                elif event_type in ("assistant", "user", "system"):
                    continue

            duration = time.time() - start_time
            logger.info(f"Claude CLI stream completed in {duration:.2f}s (model: {target.model})")

        except HTTPException as e:
            yield self._sse_error(e.status_code, e.detail)
            return
        except Exception as e:
            logger.error(f"Claude CLI stream error: {e}")
            import traceback

            logger.error(traceback.format_exc())
            yield self._sse_error(500, f"Streaming error: {str(e)}")
            return
        finally:
            if request_id and request_id in self.active_processes:
                del self.active_processes[request_id]

        # Final SSE events
        yield self._sse(
            Constants.EVENT_CONTENT_BLOCK_STOP,
            {"type": Constants.EVENT_CONTENT_BLOCK_STOP, "index": text_block_index},
        )
        yield self._sse(
            Constants.EVENT_MESSAGE_DELTA,
            {
                "type": Constants.EVENT_MESSAGE_DELTA,
                "delta": {"stop_reason": final_stop_reason, "stop_sequence": None},
                "usage": usage_data,
            },
        )
        yield self._sse(Constants.EVENT_MESSAGE_STOP, {"type": Constants.EVENT_MESSAGE_STOP})

    # ------------------------------------------------------------------
    # SSE helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sse(event_name: str, data: dict) -> str:
        return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    @staticmethod
    def _sse_error(status_code: int, message: str) -> str:
        """Emit a generic SSE error event (for internal/unexpected errors)."""
        error_event = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": f"[Claude CLI] {message}",
            },
        }
        return f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    @staticmethod
    def _sse_error_typed(error_type: str, message: str) -> str:
        """Emit an SSE error event with an Anthropic-style error type.

        This lets callers like Claude Code recognize and handle specific
        errors such as 'rate_limit_error' the same way they would for the
        real Anthropic API.
        """
        error_event = {
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
        }
        return f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel_request(self, request_id: str) -> bool:
        """Terminate a running CLI process."""
        proc = self.active_processes.get(request_id)
        if proc:
            try:
                proc.terminate()
                logger.info(f"Claude CLI process {request_id} terminated")
            except ProcessLookupError:
                pass
            return True
        return False

    def classify_cli_error(self, error_detail: Any) -> str:
        """Provide user-friendly error messages for common CLI failures."""
        error_str = str(error_detail).lower()

        if "session limit" in error_str or "rate" in error_str:
            return "Claude Code rate/session limit reached. Please try again later."
        if "invalid api key" in error_str or "fix external api key" in error_str:
            return (
                "Claude CLI auth conflict: an API key env var (ANTHROPIC_API_KEY) was "
                "inherited by the CLI. The proxy strips these for CLI calls; if you still "
                "see this, run 'claude auth' or ensure the CLI is logged in."
            )
        if "not logged in" in error_str or "auth" in error_str or "unauthorized" in error_str:
            return "Claude CLI is not authenticated. Run 'claude auth' to log in."
        if "not found" in error_str and "command" in error_str:
            return "Claude CLI not found. Check CLAUDE_CLI_PATH configuration."
        if "enoent" in error_str or "no such file" in error_str:
            return "Claude CLI executable not found. Check CLAUDE_CLI_PATH configuration."

        return str(error_detail)
