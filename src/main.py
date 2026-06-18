from fastapi import FastAPI
from src.api.endpoints import router as api_router
import uvicorn
import sys
from src.core.config import config

app = FastAPI(title="Claude-to-OpenAI API Proxy", version="1.0.0")

app.include_router(api_router)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Claude-to-OpenAI API Proxy v1.0.0")
        print("")
        print("Usage: python src/main.py")
        print("")
        print("Required environment variables:")
        print("  OPENAI_API_KEY - Your OpenAI API key")
        print("")
        print("Optional environment variables:")
        print("  ANTHROPIC_API_KEY - Expected Anthropic API key for client validation")
        print("                      If set, clients must provide this exact API key")
        print(
            f"  OPENAI_BASE_URL - OpenAI API base URL (default: https://api.openai.com/v1)"
        )
        print(f"  BIG_MODEL - Model for opus requests (default: gpt-4o)")
        print(f"  MIDDLE_MODEL - Model for sonnet requests (default: gpt-4o)")
        print(f"  SMALL_MODEL - Model for haiku requests (default: gpt-4o-mini)")
        print(f"  HOST - Server host (default: 0.0.0.0)")
        print(f"  PORT - Server port (default: 8082)")
        print(f"  LOG_LEVEL - Logging level (default: WARNING)")
        print(f"  MAX_TOKENS_LIMIT - Token limit (default: 4096)")
        print(f"  MIN_TOKENS_LIMIT - Minimum token limit (default: 100)")
        print(f"  REQUEST_TIMEOUT - Request timeout in seconds (default: 90)")
        print("")
        print("Model mapping:")
        print(f"  Claude haiku models -> {config.small_model}")
        print(f"  Claude sonnet/opus models -> {config.big_model}")
        sys.exit(0)

    # Configuration summary
    print("🚀 Claude-to-OpenAI API Proxy v1.0.0")
    print(f"✅ Configuration loaded successfully")
    print(f"   Server: {config.host}:{config.port}")
    print(f"   Client API Key Validation: {'Enabled' if config.anthropic_api_key else 'Disabled'}")
    print(f"   Max Tokens Limit: {config.max_tokens_limit}")
    print(f"   Request Timeout: {config.request_timeout}s")
    print("")
    print("📋 Model Tier Configuration:")
    print(f"   BIG (opus):")
    print(f"     Model:    {config.big_model}")
    print(f"     Base URL: {config.big_model_base_url}")
    print(f"     API Key:  {config.mask_api_key(config.big_model_api_key)}")
    print(f"   MIDDLE (sonnet):")
    print(f"     Model:    {config.middle_model}")
    print(f"     Base URL: {config.middle_model_base_url}")
    print(f"     API Key:  {config.mask_api_key(config.middle_model_api_key)}")
    print(f"   SMALL (haiku):")
    print(f"     Model:    {config.small_model}")
    print(f"     Base URL: {config.small_model_base_url}")
    print(f"     API Key:  {config.mask_api_key(config.small_model_api_key)}")
    print("")

    # Parse log level - extract just the first word to handle comments
    log_level = config.log_level.split()[0].lower()
    
    # Validate and set default if invalid
    valid_levels = ['debug', 'info', 'warning', 'error', 'critical']
    if log_level not in valid_levels:
        log_level = 'info'

    # Start server
    uvicorn.run(
        "src.main:app",
        host=config.host,
        port=config.port,
        log_level=log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
