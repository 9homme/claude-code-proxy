# Claude Code Proxy

A proxy server that lets you **mix and match LLM providers** behind a single Claude Code-compatible API. Use your **Claude Code Max subscription** (via the `claude` CLI) for heavy lifting, an **OpenAI API** for medium tasks, and a **local model** (Ollama) for lightweight work — all at the same time, transparently routed by model tier.

![Claude Code Proxy](demo.png)

## Why this proxy?

Most Claude Code proxies force you to pick **one** backend. This one lets you combine **everything**:

| Tier | What Claude Code sends | Example backend |
|------|----------------------|-----------------|
| **BIG** (opus) | Complex reasoning, architecture | 🟣 **Claude Code Max** via CLI (`opus`) |
| **MIDDLE** (sonnet) | General coding, refactoring | 🔵 **OpenAI** (`gpt-4o`) or **DeepSeek** |
| **SMALL** (haiku) | Quick lookups, completions | 🟢 **Local Ollama** (`llama3.1:8b`) or **GLM-4-Flash** |

**One proxy, three providers, zero code changes.** Claude Code just talks to `localhost:8082` and the proxy routes each request to the right backend automatically.

### What makes it special

- 🎯 **Per-tier provider routing** — Each tier (BIG/MIDDLE/SMALL) independently selects between `openai` (any OpenAI-compatible API) or `claude-cli` (your Claude Code Max subscription)
- 💰 **Maximize your subscriptions** — Use your Claude Code Max plan for opus-tier heavy reasoning while cheaper/faster providers handle the rest
- 🔌 **Any OpenAI-compatible provider** — OpenAI, Azure, DeepSeek, GLM, Ollama, vLLM, LiteLLM, and more
- 🟣 **Claude CLI integration** — Run `claude -p --model opus` under the hood; OAuth auth handled automatically
- 📡 **Full streaming support** — Real-time SSE streaming for both CLI and API backends
- 🛡️ **Anthropic-compatible errors** — Session limits return proper `rate_limit_error` (429) so Claude Code handles them gracefully
- 🔧 **Zero-code switching** — Change backends purely via environment variables; no code changes needed

## Features

- **Full Claude API Compatibility**: Complete `/v1/messages` endpoint support
- **Mixed Provider Routing**: Combine Claude Code CLI + OpenAI + local models simultaneously
- **Smart Model Mapping**: Configure BIG, MIDDLE, and SMALL models via environment variables
- **Function Calling**: Complete tool use support with proper conversion
- **Streaming Responses**: Real-time SSE streaming support for all backends
- **Image Support**: Base64 encoded image input
- **Custom Headers**: Automatic injection of custom HTTP headers for API requests
- **Error Handling**: Comprehensive error handling with Anthropic-style error types

## Quick Start

### 1. Install Dependencies

```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your API configuration
# Note: Environment variables are automatically loaded from .env file
```

### 3. Start Server

```bash
# Direct run
python start_proxy.py

# Or with UV
uv run claude-code-proxy

# Or with docker compose
docker compose up -d
```

### 4. Use with Claude Code

```bash
# If ANTHROPIC_API_KEY is not set in the proxy:
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_API_KEY="any-value" claude

# If ANTHROPIC_API_KEY is set in the proxy:
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_API_KEY="exact-matching-key" claude
```

## 🎭 Showcase: Mixed Provider Setup

Here's the killer feature — **three different providers running simultaneously** through one proxy:

```bash
# ┌─────────────────────────────────────────────────────────────────┐
# │  Claude Code  ──►  Proxy (:8082)  ──►  ┌──────────────────────┐ │
# │                                        │  BIG   → Claude CLI  │ │
# │                                        │  MID   → OpenAI API  │ │
# │                                        │  SMALL → Ollama      │ │
# │                                        └──────────────────────┘ │
# └─────────────────────────────────────────────────────────────────┘

# BIG tier: Claude Code Max subscription (opus) for complex reasoning
BIG_MODEL_PROVIDER="claude-cli"
BIG_MODEL="opus"

# MIDDLE tier: OpenAI for general coding
MIDDLE_MODEL_PROVIDER="openai"
MIDDLE_MODEL="gpt-4o"
MIDDLE_MODEL_API_KEY="sk-your-openai-key"

# SMALL tier: free local model for quick tasks
SMALL_MODEL_PROVIDER="openai"
SMALL_MODEL="qwen2.5-coder:7b"
SMALL_MODEL_BASE_URL="http://localhost:11434/v1"
SMALL_MODEL_API_KEY="dummy"
```

### How routing works

When Claude Code sends a request, the proxy inspects the requested model name and routes automatically:

```
claude-3-opus-*      → BIG tier    → claude -p --model opus    (your Max subscription)
claude-*-sonnet-*    → MIDDLE tier → OpenAI API (gpt-4o)
claude-*-haiku-*     → SMALL tier  → Ollama (qwen2.5-coder:7b)
```

**You don't need to change anything in Claude Code.** Just point it at the proxy and each request goes to the optimal backend automatically. Use your expensive Claude Max subscription where it matters (opus), cheap APIs for everyday tasks (sonnet→gpt-4o), and free local models for trivial work (haiku→ollama).

### Other popular combinations

<details>
<summary><b>🔐 Claude Max + DeepSeek + GLM-4-Flash</b></summary>

```bash
BIG_MODEL_PROVIDER="claude-cli"
BIG_MODEL="opus"

MIDDLE_MODEL_PROVIDER="openai"
MIDDLE_MODEL="deepseek-chat"
MIDDLE_MODEL_API_KEY="sk-deepseek-key"
MIDDLE_MODEL_BASE_URL="https://api.deepseek.com/v1"

SMALL_MODEL_PROVIDER="openai"
SMALL_MODEL="glm-4-flash"
SMALL_MODEL_API_KEY="sk-glm-key"
SMALL_MODEL_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
```
</details>

<details>
<summary><b>🟣 All-Claude Max (subscription only)</b></summary>

```bash
BIG_MODEL_PROVIDER="claude-cli"
BIG_MODEL="opus"
MIDDLE_MODEL_PROVIDER="claude-cli"
MIDDLE_MODEL="sonnet"
SMALL_MODEL_PROVIDER="claude-cli"
SMALL_MODEL="haiku"
```
</details>

<details>
<summary><b>🌐 LiteLLM gateway + Claude CLI</b></summary>

```bash
# Route big tasks to Claude Max, everything else through a LiteLLM gateway
BIG_MODEL_PROVIDER="claude-cli"
BIG_MODEL="opus"

MIDDLE_MODEL_PROVIDER="openai"
MIDDLE_MODEL="gpt-4o"
MIDDLE_MODEL_API_KEY="sk-litellm-key"
MIDDLE_MODEL_BASE_URL="http://your-litellm-gateway:4000/v1"

SMALL_MODEL_PROVIDER="openai"
SMALL_MODEL="claude-3-5-haiku"
SMALL_MODEL_API_KEY="sk-litellm-key"
SMALL_MODEL_BASE_URL="http://your-litellm-gateway:4000/v1"
```
</details>

## Configuration

The application automatically loads environment variables from a `.env` file in the project root using `python-dotenv`. You can also set environment variables directly in your shell.

### Environment Variables

**Required:**

- `OPENAI_API_KEY` - Your API key for the target provider

**Security:**

- `ANTHROPIC_API_KEY` - Expected Anthropic API key for client validation
  - If set, clients must provide this exact API key to access the proxy
  - If not set, any API key will be accepted

**Model Configuration:**

- `BIG_MODEL` - Model for Claude opus requests (default: `gpt-4o`)
- `MIDDLE_MODEL` - Model for Claude opus requests (default: `gpt-4o`)
- `SMALL_MODEL` - Model for Claude haiku requests (default: `gpt-4o-mini`)

**Per-Model Tier Configuration (Advanced):**

- `BIG_MODEL_API_KEY` - API key specifically for Big models
- `BIG_MODEL_BASE_URL` - Base URL specifically for Big models
- `MIDDLE_MODEL_API_KEY` - API key specifically for Middle models
- `MIDDLE_MODEL_BASE_URL` - Base URL specifically for Middle models
- `SMALL_MODEL_API_KEY` - API key specifically for Small models
- `SMALL_MODEL_BASE_URL` - Base URL specifically for Small models

*If not set, these will default to the global `OPENAI_API_KEY` and `OPENAI_BASE_URL`.*

**API Configuration:**

- `OPENAI_BASE_URL` - API base URL (default: `https://api.openai.com/v1`)

**Server Settings:**

- `HOST` - Server host (default: `0.0.0.0`)
- `PORT` - Server port (default: `8082`)
- `LOG_LEVEL` - Logging level (default: `WARNING`)

**Performance:**

- `MAX_TOKENS_LIMIT` - Token limit (default: `4096`)
- `REQUEST_TIMEOUT` - Request timeout in seconds (default: `90`)

**Custom Headers:**

- `CUSTOM_HEADER_*` - Custom headers for API requests (e.g., `CUSTOM_HEADER_ACCEPT`, `CUSTOM_HEADER_AUTHORIZATION`)
  - Uncomment in `.env` file to enable custom headers

### Custom Headers Configuration

Add custom headers to your API requests by setting environment variables with the `CUSTOM_HEADER_` prefix:

```bash
# Uncomment to enable custom headers
# CUSTOM_HEADER_ACCEPT="application/jsonstream"
# CUSTOM_HEADER_CONTENT_TYPE="application/json"
# CUSTOM_HEADER_USER_AGENT="your-app/1.0.0"
# CUSTOM_HEADER_AUTHORIZATION="Bearer your-token"
# CUSTOM_HEADER_X_API_KEY="your-api-key"
# CUSTOM_HEADER_X_CLIENT_ID="your-client-id"
# CUSTOM_HEADER_X_CLIENT_VERSION="1.0.0"
# CUSTOM_HEADER_X_REQUEST_ID="unique-request-id"
# CUSTOM_HEADER_X_TRACE_ID="trace-123"
# CUSTOM_HEADER_X_SESSION_ID="session-456"
```

### Header Conversion Rules

Environment variables with the `CUSTOM_HEADER_` prefix are automatically converted to HTTP headers:

- Environment variable: `CUSTOM_HEADER_ACCEPT`
- HTTP Header: `ACCEPT`

- Environment variable: `CUSTOM_HEADER_X_API_KEY`
- HTTP Header: `X-API-KEY`

- Environment variable: `CUSTOM_HEADER_AUTHORIZATION`
- HTTP Header: `AUTHORIZATION`

### Supported Header Types

- **Content Type**: `ACCEPT`, `CONTENT-TYPE`
- **Authentication**: `AUTHORIZATION`, `X-API-KEY`
- **Client Identification**: `USER-AGENT`, `X-CLIENT-ID`, `X-CLIENT-VERSION`
- **Tracking**: `X-REQUEST-ID`, `X-TRACE-ID`, `X-SESSION-ID`

### Usage Example

```bash
# Basic configuration
OPENAI_API_KEY="sk-your-openai-api-key-here"
OPENAI_BASE_URL="https://api.openai.com/v1"

# Enable custom headers (uncomment as needed)
CUSTOM_HEADER_ACCEPT="application/jsonstream"
CUSTOM_HEADER_CONTENT_TYPE="application/json"
CUSTOM_HEADER_USER_AGENT="my-app/1.0.0"
CUSTOM_HEADER_AUTHORIZATION="Bearer my-token"
```

The proxy will automatically include these headers in all API requests to the target LLM provider.

### Model Mapping

The proxy maps Claude model requests to your configured models:

| Claude Request                 | Mapped To     | Environment Variable   |
| ------------------------------ | ------------- | ---------------------- |
| Models with "haiku"            | `SMALL_MODEL` | Default: `gpt-4o-mini` |
| Models with "sonnet"           | `MIDDLE_MODEL`| Default: `BIG_MODEL`   |
| Models with "opus"             | `BIG_MODEL`   | Default: `gpt-4o`      |

### Provider Examples

#### OpenAI

```bash
OPENAI_API_KEY="sk-your-openai-key"
OPENAI_BASE_URL="https://api.openai.com/v1"
BIG_MODEL="gpt-4o"
MIDDLE_MODEL="gpt-4o"
SMALL_MODEL="gpt-4o-mini"
```

#### Azure OpenAI

```bash
OPENAI_API_KEY="your-azure-key"
OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/deployments/your-deployment"
BIG_MODEL="gpt-4"
MIDDLE_MODEL="gpt-4"
SMALL_MODEL="gpt-35-turbo"
```

#### Local Models (Ollama)

```bash
OPENAI_API_KEY="dummy-key"  # Required but can be dummy
OPENAI_BASE_URL="http://localhost:11434/v1"
BIG_MODEL="llama3.1:70b"
MIDDLE_MODEL="llama3.1:70b"
SMALL_MODEL="llama3.1:8b"
```

#### Per-Model Tier Configuration (Multi-Provider)

You can route different model tiers to different providers:

```bash
# Global fallback
OPENAI_API_KEY="sk-global-key"
OPENAI_BASE_URL="https://api.openai.com/v1"

# Route Big models (Opus) to a specialized provider
BIG_MODEL_API_KEY="sk-provider-a-key"
BIG_MODEL_BASE_URL="https://api.provider-a.com/v1"
BIG_MODEL="qwen-max"

# Route Small models (Haiku) to a faster/cheaper provider
SMALL_MODEL_API_KEY="sk-provider-b-key"
SMALL_MODEL_BASE_URL="https://api.provider-b.com/v1"
SMALL_MODEL="glm-4-flash"
```

#### Claude Code CLI Provider (Claude Code Max subscription)

In addition to OpenAI-compatible providers, each tier can be served by the
**Claude Code CLI** using your Claude Code Max subscription. This lets you mix
your subscription with other providers.

First, make sure the CLI is installed and authenticated:

```bash
claude auth   # log in with your Claude Code subscription
```

Then set `*_MODEL_PROVIDER="claude-cli"` for the tiers you want to route to the
CLI. The tier's `*_MODEL` value is passed to the CLI as `claude --model`, so you
can use aliases (`opus`, `sonnet`, `haiku`) or full model names.

```bash
# Route every tier to your Claude Code Max subscription
BIG_MODEL_PROVIDER="claude-cli"
BIG_MODEL="opus"
MIDDLE_MODEL_PROVIDER="claude-cli"
MIDDLE_MODEL="sonnet"
SMALL_MODEL_PROVIDER="claude-cli"
SMALL_MODEL="haiku"
```

**Mixed providers** — the main use case for this feature:

```bash
# Big tier -> Claude Code Max (via CLI)
BIG_MODEL_PROVIDER="claude-cli"
BIG_MODEL="opus"

# Middle tier -> OpenAI
MIDDLE_MODEL_PROVIDER="openai"
MIDDLE_MODEL="gpt-4o"

# Small tier -> local model
SMALL_MODEL_PROVIDER="openai"
SMALL_MODEL="llama3.1:8b"
SMALL_MODEL_BASE_URL="http://localhost:11434/v1"
```

Claude CLI backend settings:

| Variable | Default | Description |
| --- | --- | --- |
| `CLAUDE_CLI_PATH` | `claude` | Path to the `claude` executable |
| `CLAUDE_CLI_EXTRA_ARGS` | _empty_ | Comma-separated extra CLI flags |
| `CLAUDE_CLI_TIMEOUT` | `600` | Per-request timeout in seconds |
| `CLAUDE_CLI_SKIP_PERMISSIONS` | `true` | Pass `--dangerously-skip-permissions` |
| `CLAUDE_CLI_ENABLE_TOOLS` | `false` | Enable CLI built-in tools for transparent mode (see below) |

#### CLI Modes: Pure Inference vs Transparent

The CLI backend supports two modes controlled by `CLAUDE_CLI_ENABLE_TOOLS`:

**Pure inference mode** (default, `CLAUDE_CLI_ENABLE_TOOLS=false`):
- All built-in tools are disabled (`--allowedTools ""`)
- System prompt is embedded in stdin text
- Best for: using the CLI as a text-only LLM backend alongside other providers

**Transparent mode** (`CLAUDE_CLI_ENABLE_TOOLS=true`):
- CLI keeps all built-in tools (Bash, Edit, Read, etc.)
- System prompt is passed via `--append-system-prompt` (proper system role)
- Claude Code works seamlessly through the proxy as if calling the real API
- Best for: full agentic behavior, letting Claude Code use its native tools

```bash
# Transparent mode — Claude Code works seamlessly via CLI
BIG_MODEL_PROVIDER="claude-cli"
BIG_MODEL="opus"
CLAUDE_CLI_ENABLE_TOOLS="true"
```

#### Other Providers

Any OpenAI-compatible API can be used by setting the appropriate `OPENAI_BASE_URL`.

## Usage Examples

### Basic Chat

```python
import httpx

response = httpx.post(
    "http://localhost:8082/v1/messages",
    json={
        "model": "claude-3-5-sonnet-20241022",  # Maps to MIDDLE_MODEL
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
)
```

## Integration with Claude Code

This proxy is designed to work seamlessly with Claude Code CLI:

```bash
# Start the proxy
python start_proxy.py

# Use Claude Code with the proxy
ANTHROPIC_BASE_URL=http://localhost:8082 claude

# Or set permanently
export ANTHROPIC_BASE_URL=http://localhost:8082
claude
```

## Testing

Test proxy functionality:

```bash
# Run comprehensive tests
python src/test_claude_to_openai.py
```

## Development

### Using UV

```bash
# Install dependencies
uv sync

# Run server
uv run claude-code-proxy

# Format code
uv run black src/
uv run isort src/

# Type checking
uv run mypy src/
```

### Project Structure

```
claude-code-proxy/
├── src/
│   ├── main.py                     # Main server
│   ├── test_claude_to_openai.py    # Tests
│   └── [other modules...]
├── start_proxy.py                  # Startup script
├── .env.example                    # Config template
└── README.md                       # This file
```

## Performance

- **Async/await** for high concurrency
- **Connection pooling** for efficiency
- **Streaming support** for real-time responses
- **Configurable timeouts** and retries
- **Smart error handling** with detailed logging

## License

MIT License
