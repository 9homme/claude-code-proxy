from typing import Tuple
from src.core.config import config

class ModelManager:
    def __init__(self, config):
        self.config = config
    
    def get_model_config(self, claude_model: str) -> Tuple[str, str, str]:
        """
        Map Claude model to OpenAI model and return its configuration.
        Returns: (openai_model_name, api_key, base_url)
        """
        # If it's already an OpenAI model, return as-is with default config
        if claude_model.startswith("gpt-") or claude_model.startswith("o1-"):
            return claude_model, self.config.openai_api_key, self.config.openai_base_url

        # If it's other supported models (ARK/Doubao/DeepSeek), return as-is
        if (claude_model.startswith("ep-") or claude_model.startswith("doubao-") or 
            claude_model.startswith("deepseek-")):
            return claude_model, self.config.openai_api_key, self.config.openai_base_url
        
        # Map based on model naming patterns
        model_lower = claude_model.lower()
        if 'haiku' in model_lower:
            return self.config.small_model, self.config.small_model_api_key, self.config.small_model_base_url
        elif 'sonnet' in model_lower:
            return self.config.middle_model, self.config.middle_model_api_key, self.config.middle_model_base_url
        elif 'opus' in model_lower:
            return self.config.big_model, self.config.big_model_api_key, self.config.big_model_base_url
        else:
            # Default to big model for unknown models
            return self.config.big_model, self.config.big_model_api_key, self.config.big_model_base_url

    def map_claude_model_to_openai(self, claude_model: str) -> str:
        """Map Claude model names to OpenAI model names based on BIG/SMALL pattern"""
        model_name, _, _ = self.get_model_config(claude_model)
        return model_name

model_manager = ModelManager(config)