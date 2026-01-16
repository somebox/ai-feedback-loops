"""Shared pytest fixtures for imageloop tests."""

import pytest
from pathlib import Path
from PIL import Image
import io
import base64


@pytest.fixture
def fixtures_dir() -> Path:
    """Directory containing test fixture images."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tiny_image(fixtures_dir: Path) -> Path:
    """64x64 minimum size test image."""
    img_path = fixtures_dir / "tiny_square.png"
    if not img_path.exists():
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def landscape_image(fixtures_dir: Path) -> Path:
    """400x300 landscape test image (4:3 aspect ratio)."""
    img_path = fixtures_dir / "landscape_4x3.png"
    if not img_path.exists():
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (400, 300), color=(0, 128, 255))
        # Add a simple gradient
        pixels = img.load()
        for y in range(300):
            for x in range(400):
                pixels[x, y] = (x % 256, (x + y) % 256, y % 256)
        img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def portrait_image(fixtures_dir: Path) -> Path:
    """300x400 portrait test image (3:4 aspect ratio)."""
    img_path = fixtures_dir / "portrait_3x4.png"
    if not img_path.exists():
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (300, 400), color=(255, 128, 0))
        # Add a simple gradient
        pixels = img.load()
        for y in range(400):
            for x in range(300):
                pixels[x, y] = (y % 256, (x + y) % 256, x % 256)
        img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def wide_image(fixtures_dir: Path) -> Path:
    """640x360 wide test image (16:9 aspect ratio)."""
    img_path = fixtures_dir / "wide_16x9.png"
    if not img_path.exists():
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (640, 360), color=(0, 255, 128))
        # Add a simple gradient
        pixels = img.load()
        for y in range(360):
            for x in range(640):
                pixels[x, y] = (x % 256, (x * 2 + y) % 256, (y * 2) % 256)
        img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def oversized_image(fixtures_dir: Path) -> Path:
    """3000x2000 image that needs downscaling."""
    img_path = fixtures_dir / "oversized.png"
    if not img_path.exists():
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (3000, 2000), color=(128, 128, 128))
        # Add a simple pattern
        pixels = img.load()
        for y in range(2000):
            for x in range(3000):
                if (x + y) % 100 < 50:
                    pixels[x, y] = (255, 255, 255)
                else:
                    pixels[x, y] = (0, 0, 0)
        img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def odd_ratio_image(fixtures_dir: Path) -> Path:
    """500x333 image with non-standard aspect ratio."""
    img_path = fixtures_dir / "odd_ratio.png"
    if not img_path.exists():
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (500, 333), color=(200, 100, 50))
        # Add a simple gradient
        pixels = img.load()
        for y in range(333):
            for x in range(500):
                pixels[x, y] = ((x * 3) % 256, (y * 3) % 256, ((x + y) * 2) % 256)
        img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def sample_settings() -> dict:
    """Minimal settings dict for testing."""
    return {
        "models": {
            "flux-pro": "black-forest-labs/flux.2-pro",
            "flux-klein": "black-forest-labs/flux.2-klein-4b",
        },
        "prompts": {
            "up": "Gently pan the camera up, extending the image.",
            "next": "show what happens moments later in this scene",
        },
        "defaults": {
            "model": "flux-pro",
            "frames": 10,
            "fps": 1,
            "temperature": 0.7,
            "top_p": 0.9,
            "size": "auto",
            "output_format": "mp4",
            "output_dir": "output",
        },
        "sizes": {
            "square": [1024, 1024],
            "landscape": [1024, 768],
            "portrait": [768, 1024],
            "wide": [1280, 720],
            "tall": [720, 1280],
        },
        "api": {
            "timeout_seconds": 120,
            "max_image_dimension": 2048,
        },
    }
