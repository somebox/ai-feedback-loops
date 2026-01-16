"""Tests for imageloop.settings module - configuration loading."""

import yaml
from pathlib import Path
import pytest


def test_settings_load_from_yaml(tmp_path):
    """Settings load from a yaml file."""
    from imageloop import settings
    
    # Create a test settings file
    test_settings = {
        "models": {
            "test-model": "test/model-id"
        },
        "prompts": {
            "test-mode": "test prompt"
        },
        "defaults": {
            "model": "test-model",
            "frames": 5
        },
        "sizes": {
            "square": [512, 512]
        },
        "api": {
            "timeout_seconds": 60
        }
    }
    
    settings_file = tmp_path / "test_settings.yaml"
    with open(settings_file, "w") as f:
        yaml.dump(test_settings, f)
    
    # Load settings
    loaded = settings.load_settings(settings_file)
    
    assert loaded["models"]["test-model"] == "test/model-id"
    assert loaded["prompts"]["test-mode"] == "test prompt"
    assert loaded["defaults"]["model"] == "test-model"


def test_settings_fallback_to_defaults(tmp_path):
    """Missing yaml uses embedded defaults."""
    from imageloop import settings
    
    # Create a non-existent path in tmp_dir
    non_existent = tmp_path / "does_not_exist.yaml"
    # Ensure it doesn't exist
    assert not non_existent.exists()
    
    # Should use defaults
    loaded = settings.load_settings(non_existent)
    
    # Should have all required sections
    assert "models" in loaded
    assert "prompts" in loaded
    assert "defaults" in loaded
    assert "sizes" in loaded
    assert "api" in loaded
    
    # Should have expected defaults
    assert loaded["defaults"]["model"] == "flux-pro"
    assert "flux-pro" in loaded["models"]


def test_get_model_resolves_shortcut():
    """'flux-pro' resolves to full model ID."""
    from imageloop import settings
    
    # Load default settings
    cfg = settings.load_settings()
    
    model_id = settings.get_model("flux-pro", cfg)
    assert model_id == "black-forest-labs/flux.2-pro"


def test_get_model_passes_through_full_id():
    """Full model IDs pass through unchanged."""
    from imageloop import settings
    
    cfg = settings.load_settings()
    
    full_id = "some-provider/model-name"
    model_id = settings.get_model(full_id, cfg)
    assert model_id == full_id


def test_get_prompt_returns_preset():
    """Mode name returns corresponding prompt text."""
    from imageloop import settings
    
    cfg = settings.load_settings()
    
    prompt = settings.get_prompt("up", None, cfg)
    assert "pan the camera up" in prompt.lower()


def test_get_prompt_custom_mode():
    """'custom' mode requires explicit prompt."""
    from imageloop import settings
    
    cfg = settings.load_settings()
    
    # With custom prompt provided
    prompt = settings.get_prompt("custom", "My custom prompt", cfg)
    assert prompt == "My custom prompt"
    
    # Without custom prompt
    with pytest.raises(ValueError) as exc_info:
        settings.get_prompt("custom", None, cfg)
    
    assert "custom prompt" in str(exc_info.value).lower()


def test_unknown_mode_raises():
    """Unknown mode raises ValueError with available modes."""
    from imageloop import settings
    
    cfg = settings.load_settings()
    
    with pytest.raises(ValueError) as exc_info:
        settings.get_prompt("unknown-mode", None, cfg)
    
    error_msg = str(exc_info.value).lower()
    assert "unknown" in error_msg or "available" in error_msg


def test_get_model_short_name_from_id():
    """get_model_short_name converts full model ID to shortcut."""
    from imageloop import settings
    
    cfg = settings.load_settings()
    
    # Known model
    short_name = settings.get_model_short_name("black-forest-labs/flux.2-pro", cfg)
    assert short_name == "flux-pro"
    
    # Unknown model - should sanitize
    unknown_id = "some-provider/model.name.v2"
    short_name = settings.get_model_short_name(unknown_id, cfg)
    assert "/" not in short_name
    assert "." not in short_name
    assert len(short_name) <= 20
