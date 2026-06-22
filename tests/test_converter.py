import pytest
from unittest.mock import patch
from src.core.config import Config
from src.core.constants import Constants
from src.models.claude import (
    ClaudeMessage,
    ClaudeContentBlockText,
    ClaudeContentBlockImage,
    ClaudeContentBlockThinking,
    ClaudeContentBlockRedactedThinking,
)
from src.conversion.request_converter import (
    convert_claude_user_message,
    convert_claude_assistant_message,
)

def test_config_defaults():
    """Test default values of the newly added configs."""
    config = Config()
    assert config.strip_images is False
    assert config.strip_image_models == ["qwen", "deepseek"]

def test_convert_user_message_text_only():
    """Test user message conversion with text content only."""
    msg = ClaudeMessage(
        role="user",
        content=[ClaudeContentBlockText(type="text", text="Hello world")]
    )
    result = convert_claude_user_message(msg)
    assert result == {"role": "user", "content": "Hello world"}

def test_convert_user_message_thinking():
    """Test user message conversion with thinking blocks."""
    msg = ClaudeMessage(
        role="user",
        content=[
            ClaudeContentBlockThinking(
                type="thinking",
                thinking="I should greet the user",
                signature="sig"
            ),
            ClaudeContentBlockText(type="text", text="Hello!")
        ]
    )
    result = convert_claude_user_message(msg)
    expected_content = "<thinking>\nI should greet the user\n</thinking>\n\nHello!"
    assert result == {"role": "user", "content": expected_content}

def test_convert_user_message_redacted_thinking():
    """Test user message conversion with redacted thinking blocks."""
    msg = ClaudeMessage(
        role="user",
        content=[
            ClaudeContentBlockRedactedThinking(
                type="redacted_thinking",
                data="redacted-data"
            ),
            ClaudeContentBlockText(type="text", text="Hello!")
        ]
    )
    result = convert_claude_user_message(msg)
    expected_content = "<thinking>\n[Redacted Thinking: redacted-data]\n</thinking>\n\nHello!"
    assert result == {"role": "user", "content": expected_content}

def test_convert_user_message_with_image_no_stripping():
    """Test image conversion when stripping is disabled."""
    msg = ClaudeMessage(
        role="user",
        content=[
            ClaudeContentBlockText(type="text", text="Look at this:"),
            ClaudeContentBlockImage(
                type="image",
                source={
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "abcdef"
                }
            )
        ]
    )
    
    with patch("src.core.config.config.strip_images", False), \
         patch("src.core.config.config.strip_image_models", ["qwen", "deepseek"]):
        result = convert_claude_user_message(msg, target_model="gpt-4o")
        
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2
        assert result["content"][0] == {"type": "text", "text": "Look at this:"}
        assert result["content"][1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abcdef"}
        }

def test_convert_user_message_with_image_global_stripping():
    """Test image conversion when global stripping is enabled."""
    msg = ClaudeMessage(
        role="user",
        content=[
            ClaudeContentBlockText(type="text", text="Look at this:"),
            ClaudeContentBlockImage(
                type="image",
                source={
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "abcdef"
                }
            )
        ]
    )
    
    with patch("src.core.config.config.strip_images", True), \
         patch("src.core.config.config.strip_image_models", ["qwen", "deepseek"]):
        result = convert_claude_user_message(msg, target_model="gpt-4o")
        
        # When image is stripped, only text remains, so content should be a string
        assert result == {"role": "user", "content": "Look at this:"}

def test_convert_user_message_with_image_model_stripping():
    """Test image conversion when stripping is triggered by model name."""
    msg = ClaudeMessage(
        role="user",
        content=[
            ClaudeContentBlockText(type="text", text="Look at this:"),
            ClaudeContentBlockImage(
                type="image",
                source={
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "abcdef"
                }
            )
        ]
    )
    
    with patch("src.core.config.config.strip_images", False), \
         patch("src.core.config.config.strip_image_models", ["qwen", "deepseek"]):
        # Model match 'qwen-vl' (contains 'qwen')
        result_qwen = convert_claude_user_message(msg, target_model="qwen-vl")
        assert result_qwen == {"role": "user", "content": "Look at this:"}
        
        # Model match 'deepseek-chat' (contains 'deepseek')
        result_deepseek = convert_claude_user_message(msg, target_model="deepseek-chat")
        assert result_deepseek == {"role": "user", "content": "Look at this:"}
        
        # Model no match 'gpt-4o'
        result_gpt = convert_claude_user_message(msg, target_model="gpt-4o")
        assert isinstance(result_gpt["content"], list)
        assert len(result_gpt["content"]) == 2

def test_convert_assistant_message_thinking():
    """Test assistant message conversion with thinking blocks."""
    msg = ClaudeMessage(
        role="assistant",
        content=[
            ClaudeContentBlockThinking(
                type="thinking",
                thinking="Let's formulate the reply.",
                signature="sig"
            ),
            ClaudeContentBlockText(type="text", text="Hello back!")
        ]
    )
    result = convert_claude_assistant_message(msg)
    expected_content = "<thinking>\nLet's formulate the reply.\n</thinking>Hello back!"
    assert result == {"role": "assistant", "content": expected_content}
