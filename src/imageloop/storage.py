"""Image file I/O and data URI operations."""

import base64
import io
import os
from pathlib import Path

import yaml
from PIL import Image


def load_api_key() -> str:
    """Load OpenRouter API key from environment or secrets.yaml."""
    # Check environment first
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return api_key

    # Try secrets.yaml in project root
    secrets_path = Path(__file__).parent.parent.parent / "secrets.yaml"
    if secrets_path.exists():
        with open(secrets_path) as f:
            secrets = yaml.safe_load(f)
            api_key = secrets.get("openrouter_api_key")
            if api_key:
                return api_key

    raise ValueError(
        "No API key found. Set OPENROUTER_API_KEY env var or add to secrets.yaml"
    )


def image_to_data_uri(file_path: str | Path) -> str:
    """Read an image file and return as a data URI."""
    image = Image.open(file_path)

    # Convert to RGB if necessary (for PNG with alpha, etc.)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    output_buffer = io.BytesIO()
    image.save(output_buffer, format="PNG")
    image_data = output_buffer.getvalue()

    encoded_string = base64.b64encode(image_data).decode("utf-8")
    return f"data:image/png;base64,{encoded_string}"


def data_uri_to_image(data_uri: str) -> Image.Image:
    """Convert a data URI to a PIL Image."""
    _, encoded = data_uri.split(",", 1)
    image_data = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_data))


def save_data_uri(data_uri: str, output_path: Path) -> int:
    """Save a data URI as a PNG file. Returns file size in bytes."""
    _, encoded = data_uri.split(",", 1)
    image_data = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_data))
    file_path = output_path.with_suffix(".png")
    image.save(file_path, format="PNG", optimize=False)
    return len(image_data)


def rescale_image(data_uri: str, max_dimension: int = 2048) -> str:
    """Rescale image to fit within max dimension while preserving aspect ratio."""
    image = data_uri_to_image(data_uri)

    # Only rescale if needed
    width, height = image.size
    if width <= max_dimension and height <= max_dimension:
        return data_uri

    image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    output_buffer = io.BytesIO()
    image.save(output_buffer, format="PNG", optimize=False)

    encoded_string = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded_string}"
