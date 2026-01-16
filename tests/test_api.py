"""Tests for imageloop.api module - OpenRouter API client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from imageloop import storage


# Mocked tests (no API calls)
@pytest.mark.asyncio
async def test_parse_image_from_message_content(landscape_image):
    """Image extracted from message.content.output_image."""
    from imageloop import api, storage
    
    data_uri = storage.image_to_data_uri(landscape_image)
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_image",
                        "image_url": data_uri,
                    }
                ],
            }
        ],
        "usage": {"cost": 0.01, "total_tokens": 100},
        "id": "test-id",
    }
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        result = await api.generate_image(
            prompt="test",
            image_data_uri=data_uri,
            model="test-model",
            api_key="test-key",
        )
        
        assert result["success"] is True
        assert result["image"] == data_uri


@pytest.mark.asyncio
async def test_parse_image_from_generation_call(landscape_image):
    """Image extracted from image_generation_call.result."""
    from imageloop import api, storage
    
    data_uri = storage.image_to_data_uri(landscape_image)
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "output": [
            {
                "type": "image_generation_call",
                "result": data_uri,
            }
        ],
        "usage": {"cost": 0.01},
    }
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        result = await api.generate_image(
            prompt="test",
            image_data_uri=data_uri,
            model="test-model",
            api_key="test-key",
        )
        
        assert result["success"] is True
        assert result["image"] == data_uri


@pytest.mark.asyncio
async def test_parse_error_response(landscape_image):
    """API error returns failure with message."""
    from imageloop import api, storage
    
    data_uri = storage.image_to_data_uri(landscape_image)
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "error": {"message": "Content policy violation"},
        "usage": {"cost": 0.0},
    }
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        result = await api.generate_image(
            prompt="test",
            image_data_uri=data_uri,
            model="test-model",
            api_key="test-key",
        )
        
        assert result["success"] is False
        assert result["error"] == "Content policy violation"


@pytest.mark.asyncio
async def test_parse_no_image_in_response(landscape_image):
    """No image in response returns failure."""
    from imageloop import api, storage
    
    data_uri = storage.image_to_data_uri(landscape_image)
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "output": [{"type": "text", "content": "Some text output"}],
        "usage": {"cost": 0.01},
    }
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        result = await api.generate_image(
            prompt="test",
            image_data_uri=data_uri,
            model="test-model",
            api_key="test-key",
        )
        
        assert result["success"] is False
        assert "No image" in result.get("error", "") or "error" in result


@pytest.mark.asyncio
async def test_fetch_image_url_returns_data_uri():
    """fetch_image_url downloads image and converts to data URI."""
    from imageloop.api import fetch_image_url
    
    # Mock HTTP response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fake_image_data"
    mock_response.headers = {"content-type": "image/png"}
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value.get.return_value.raise_for_status = MagicMock()
        
        result = await fetch_image_url("https://example.com/image.png")
        
        assert result.startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_generate_image_returns_result_dict(landscape_image):
    """generate_image returns dict with success, image, usage keys."""
    from imageloop import api
    
    # Create a test data URI
    data_uri = storage.image_to_data_uri(landscape_image)
    
    # Mock the API call
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_image",
                        "image_url": data_uri,
                    }
                ],
            }
        ],
        "usage": {"cost": 0.01, "total_tokens": 100},
        "id": "test-id",
        "model": "test-model",
    }
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        result = await api.generate_image(
            prompt="test prompt",
            image_data_uri=data_uri,
            model="test-model",
            api_key="test-key",
        )
        
        assert "success" in result
        assert "image" in result
        assert "usage" in result
        assert "response" in result
        assert "duration" in result


# Live API test (uses cheap model) - marked to skip unless explicitly run
@pytest.mark.live_api
@pytest.mark.asyncio
async def test_live_generate_image(landscape_image):
    """Live API call generates an image (flux-klein model)."""
    from imageloop import api, storage
    
    # Skip if no API key
    try:
        api_key = storage.load_api_key()
    except ValueError:
        pytest.skip("No API key available for live test")
    
    # Create a test data URI
    data_uri = storage.image_to_data_uri(landscape_image)
    
    # Use cheap flux-klein model
    result = await api.generate_image(
        prompt="make this image slightly brighter",
        image_data_uri=data_uri,
        model="black-forest-labs/flux.2-klein-4b",
        api_key=api_key,
    )
    
    # Should succeed
    assert result["success"] is True
    assert result["image"] is not None
    assert result["image"].startswith("data:image")
    assert result["usage"] is not None
