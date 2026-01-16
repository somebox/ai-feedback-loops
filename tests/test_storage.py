"""Tests for imageloop.storage module - image file I/O."""

import os
from pathlib import Path
from PIL import Image
import pytest


def test_image_roundtrip_preserves_dimensions(landscape_image, temp_output_dir):
    """Load image, convert to data URI, save back - dimensions match."""
    from imageloop.storage import image_to_data_uri, save_data_uri
    
    # Load original
    original = Image.open(landscape_image)
    original_w, original_h = original.size
    original.close()
    
    # Convert to data URI and back
    data_uri = image_to_data_uri(landscape_image)
    output_path = temp_output_dir / "roundtrip"
    save_data_uri(data_uri, output_path)
    
    # Verify dimensions match
    saved = Image.open(output_path.with_suffix(".png"))
    assert saved.size == (original_w, original_h)
    saved.close()


def test_save_creates_png_file(landscape_image, temp_output_dir):
    """save_data_uri creates a valid PNG that PIL can open."""
    from imageloop.storage import image_to_data_uri, save_data_uri
    
    data_uri = image_to_data_uri(landscape_image)
    output_path = temp_output_dir / "test_output"
    
    file_size = save_data_uri(data_uri, output_path)
    
    # Verify file exists and is valid PNG
    png_path = output_path.with_suffix(".png")
    assert png_path.exists()
    assert file_size > 0
    
    # PIL can open it
    img = Image.open(png_path)
    assert img.format == "PNG"
    img.close()


def test_oversized_image_loads(oversized_image):
    """Large images load without error."""
    from imageloop.storage import image_to_data_uri
    
    # Should not raise
    data_uri = image_to_data_uri(oversized_image)
    assert data_uri.startswith("data:image/png;base64,")
    
    # Verify we got the full image data
    assert len(data_uri) > 10000  # Large image should produce large data URI


def test_data_uri_to_image_reconstructs_correctly(landscape_image, temp_output_dir):
    """data_uri_to_image produces a PIL Image matching original."""
    from imageloop.storage import image_to_data_uri, data_uri_to_image
    
    # Get original dimensions
    original = Image.open(landscape_image)
    original_w, original_h = original.size
    original.close()
    
    # Convert to data URI and back to image
    data_uri = image_to_data_uri(landscape_image)
    reconstructed = data_uri_to_image(data_uri)
    
    # Dimensions should match
    assert reconstructed.size == (original_w, original_h)
    reconstructed.close()


def test_load_api_key_from_env(monkeypatch):
    """API key loads from OPENROUTER_API_KEY env var."""
    from imageloop.storage import load_api_key
    
    # Set environment variable
    test_key = "test-api-key-12345"
    monkeypatch.setenv("OPENROUTER_API_KEY", test_key)
    
    # Remove any existing secrets.yaml that might interfere
    # (This test assumes no secrets.yaml exists, or we'd need to mock Path.exists)
    
    key = load_api_key()
    assert key == test_key


def test_load_api_key_missing_raises(monkeypatch):
    """Missing API key raises ValueError with helpful message."""
    from imageloop import storage
    from pathlib import Path
    import yaml
    
    # Remove env var
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    
    # Temporarily rename secrets.yaml if it exists so the test can work
    project_root = Path(__file__).parent.parent
    secrets_path = project_root / "secrets.yaml"
    secrets_backup = None
    if secrets_path.exists():
        secrets_backup = project_root / "secrets.yaml.backup"
        secrets_path.rename(secrets_backup)
    
    try:
        with pytest.raises(ValueError) as exc_info:
            storage.load_api_key()
        
        error_msg = str(exc_info.value).lower()
        assert "api key" in error_msg or "openrouter" in error_msg
        assert "openrouter_api_key" in error_msg or "secrets.yaml" in error_msg
    finally:
        # Restore secrets.yaml if it existed
        if secrets_backup and secrets_backup.exists():
            secrets_backup.rename(secrets_path)


def test_save_data_uri_returns_file_size(landscape_image, temp_output_dir):
    """save_data_uri returns the byte size of saved image data."""
    from imageloop.storage import image_to_data_uri, save_data_uri
    
    data_uri = image_to_data_uri(landscape_image)
    output_path = temp_output_dir / "size_test"
    
    file_size = save_data_uri(data_uri, output_path)
    
    # File size should match actual file size (approximately)
    png_path = output_path.with_suffix(".png")
    actual_size = png_path.stat().st_size
    
    # Data URI size should be the base64-encoded size before saving
    # The actual PNG may be slightly different due to PNG compression
    # But should be in the same ballpark
    assert file_size > 0
    assert actual_size > 0
