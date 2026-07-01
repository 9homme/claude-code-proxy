import os
import sys

# Configuration
class Config:
    def __init__(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Add Anthropic API key for client validation
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            print("Warning: ANTHROPIC_API_KEY not set. Client API key validation will be disabled.")
        
        self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.azure_api_version = os.environ.get("AZURE_API_VERSION")  # For Azure OpenAI
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8082"))
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "4096"))
        self.min_tokens_limit = int(os.environ.get("MIN_TOKENS_LIMIT", "100"))
        
        self.request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "90"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "2"))
        
        # Multimodal settings
        self.strip_images = os.environ.get("STRIP_IMAGES", "false").lower() in ("true", "1", "yes")
        
        # Strip images based on model name substring
        models_str = os.environ.get("STRIP_IMAGE_MODELS", "qwen,deepseek")
        self.strip_image_models = [m.strip().lower() for m in models_str.split(",") if m.strip()]
        
        # Model settings - BIG, MIDDLE and SMALL models
        self.big_model = os.environ.get("BIG_MODEL", "gpt-4o")
        self.middle_model = os.environ.get("MIDDLE_MODEL", self.big_model)
        self.small_model = os.environ.get("SMALL_MODEL", "gpt-4o-mini")

        # Per-model configuration
        self.big_model_api_key = os.environ.get("BIG_MODEL_API_KEY", self.openai_api_key)
        self.big_model_base_url = os.environ.get("BIG_MODEL_BASE_URL", self.openai_base_url)

        self.middle_model_api_key = os.environ.get("MIDDLE_MODEL_API_KEY", self.openai_api_key)
        self.middle_model_base_url = os.environ.get("MIDDLE_MODEL_BASE_URL", self.openai_base_url)

        self.small_model_api_key = os.environ.get("SMALL_MODEL_API_KEY", self.openai_api_key)
        self.small_model_base_url = os.environ.get("SMALL_MODEL_BASE_URL", self.openai_base_url)

        # Per-model provider selection ("openai" or "claude-cli")
        self.big_model_provider = os.environ.get("BIG_MODEL_PROVIDER", "openai").lower()
        self.middle_model_provider = os.environ.get("MIDDLE_MODEL_PROVIDER", "openai").lower()
        self.small_model_provider = os.environ.get("SMALL_MODEL_PROVIDER", "openai").lower()

        # Claude CLI backend settings
        self.claude_cli_path = os.environ.get("CLAUDE_CLI_PATH", "claude")
        self.claude_cli_extra_args = [
            a.strip() for a in os.environ.get("CLAUDE_CLI_EXTRA_ARGS", "").split(",") if a.strip()
        ]
        self.claude_cli_timeout = int(os.environ.get("CLAUDE_CLI_TIMEOUT", "600"))
        self.claude_cli_skip_permissions = os.environ.get(
            "CLAUDE_CLI_SKIP_PERMISSIONS", "true"
        ).lower() in ("true", "1", "yes")
        
    def validate_api_key(self):
        """Basic API key validation"""
        if not self.openai_api_key:
            return False
        # Basic format check for OpenAI API keys
        if not self.openai_api_key.startswith('sk-'):
            return False
        return True
        
    def validate_client_api_key(self, client_api_key):
        """Validate client's Anthropic API key"""
        # If no ANTHROPIC_API_KEY is set in environment, skip validation
        if not self.anthropic_api_key:
            return True
            
        # Check if the client's API key matches the expected value
        return client_api_key == self.anthropic_api_key
    
    def get_custom_headers(self):
        """Get custom headers from environment variables"""
        custom_headers = {}
        
        # Get all environment variables
        env_vars = dict(os.environ)
        
        # Find CUSTOM_HEADER_* environment variables
        for env_key, env_value in env_vars.items():
            if env_key.startswith('CUSTOM_HEADER_'):
                # Convert CUSTOM_HEADER_KEY to Header-Key
                # Remove 'CUSTOM_HEADER_' prefix and convert to header format
                header_name = env_key[14:]  # Remove 'CUSTOM_HEADER_' prefix
                
                if header_name:  # Make sure it's not empty
                    # Convert underscores to hyphens for HTTP header format
                    header_name = header_name.replace('_', '-')
                    custom_headers[header_name] = env_value
        
        return custom_headers

    def mask_api_key(self, api_key: str) -> str:
        """Mask API key for logging"""
        if not api_key:
            return "Not set"
        if len(api_key) <= 8:
            return "*" * len(api_key)
        return f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}"

try:
    config = Config()
    print(f" Configuration loaded: API_KEY={'*' * 20}..., BASE_URL='{config.openai_base_url}'")
except Exception as e:
    print(f"=4 Configuration Error: {e}")
    sys.exit(1)
