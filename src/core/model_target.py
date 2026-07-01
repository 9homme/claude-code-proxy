"""Model target descriptor used to route requests to a backend provider."""

from dataclasses import dataclass
from typing import Optional


PROVIDER_OPENAI = "openai"
PROVIDER_CLAUDE_CLI = "claude-cli"
VALID_PROVIDERS = {PROVIDER_OPENAI, PROVIDER_CLAUDE_CLI}


@dataclass
class ModelTarget:
    """Describes where a request for a given Claude model should be routed.

    For the `openai` provider, `model`/`api_key`/`base_url` describe the
    OpenAI-compatible endpoint to call.

    For the `claude-cli` provider, `model` holds the Claude model alias/name
    (e.g. "opus", "sonnet", "claude-sonnet-4-5") to pass to the CLI via
    `--model`, and `api_key`/`base_url` are unused.
    """

    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    @property
    def is_claude_cli(self) -> bool:
        return self.provider == PROVIDER_CLAUDE_CLI

    @property
    def is_openai(self) -> bool:
        return self.provider == PROVIDER_OPENAI