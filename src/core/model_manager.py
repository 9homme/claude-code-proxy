from typing import Tuple
from src.core.config import config
from src.core.model_target import ModelTarget, PROVIDER_OPENAI


class ModelManager:
    def __init__(self, config):
        self.config = config

    def _tier_for_claude_model(self, claude_model: str) -> str:
        """Return the tier name ("big", "middle", "small") for a Claude model.

        Mirrors the original mapping logic: haiku -> small, sonnet -> middle,
        everything else (including opus and unknown) -> big.
        """
        model_lower = claude_model.lower()
        if "haiku" in model_lower:
            return "small"
        if "sonnet" in model_lower:
            return "middle"
        # opus and unknown models default to big
        return "big"

    def _target_for_openai_passthrough(self, claude_model: str) -> ModelTarget:
        """Handle models that are already OpenAI/other provider model names.

        These are passed through using the global OpenAI credentials, matching
        the previous behavior where such models bypassed tier mapping.
        """
        return ModelTarget(
            provider=PROVIDER_OPENAI,
            model=claude_model,
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url,
        )

    def _target_for_tier(self, tier: str) -> ModelTarget:
        """Build a ModelTarget for a tier using that tier's provider config."""
        if tier == "small":
            return ModelTarget(
                provider=self.config.small_model_provider,
                model=self.config.small_model,
                api_key=self.config.small_model_api_key,
                base_url=self.config.small_model_base_url,
            )
        if tier == "middle":
            return ModelTarget(
                provider=self.config.middle_model_provider,
                model=self.config.middle_model,
                api_key=self.config.middle_model_api_key,
                base_url=self.config.middle_model_base_url,
            )
        # big
        return ModelTarget(
            provider=self.config.big_model_provider,
            model=self.config.big_model,
            api_key=self.config.big_model_api_key,
            base_url=self.config.big_model_base_url,
        )

    def get_model_config(self, claude_model: str) -> ModelTarget:
        """Resolve a Claude model name to its routing target.

        For OpenAI/other provider model names (gpt-*, o1-*, ep-*, doubao-*,
        deepseek-*) we pass through with global OpenAI credentials. For Claude
        model names we route by tier (big/middle/small), each of which may be
        configured to use the OpenAI or claude-cli backend.
        """
        # If it's already an OpenAI model, return as-is with default config
        if claude_model.startswith("gpt-") or claude_model.startswith("o1-"):
            return self._target_for_openai_passthrough(claude_model)

        # If it's other supported models (ARK/Doubao/DeepSeek), return as-is
        if (
            claude_model.startswith("ep-")
            or claude_model.startswith("doubao-")
            or claude_model.startswith("deepseek-")
        ):
            return self._target_for_openai_passthrough(claude_model)

        tier = self._tier_for_claude_model(claude_model)
        return self._target_for_tier(tier)

    def map_claude_model_to_openai(self, claude_model: str) -> str:
        """Map Claude model names to the target model name.

        Kept for backward compatibility with the request converter. Returns the
        configured model name for the resolved tier regardless of provider.
        """
        return self.get_model_config(claude_model).model


model_manager = ModelManager(config)