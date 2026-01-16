"""Tests for imageloop.job module - job orchestration."""

import glob
from pathlib import Path
from PIL import Image
import pytest
from imageloop import storage


def test_find_last_frame_finds_highest_number(temp_output_dir):
    """find_last_frame returns the highest frame number."""
    from imageloop.job import find_last_frame
    
    images_dir = temp_output_dir / "images"
    images_dir.mkdir()
    
    # Create some frame files
    for i in [0, 1, 3, 5]:
        frame_path = images_dir / f"frame_{i:03d}.png"
        img = Image.new("RGB", (100, 100), color=(i * 50, 0, 0))
        img.save(frame_path)
    
    last_num, last_path = find_last_frame(images_dir)
    
    assert last_num == 5
    assert last_path.name == "frame_005.png"


def test_find_last_frame_returns_none_if_empty(temp_output_dir):
    """find_last_frame returns None if no frames found."""
    from imageloop.job import find_last_frame
    
    images_dir = temp_output_dir / "images"
    images_dir.mkdir()
    
    result = find_last_frame(images_dir)
    assert result is None


def test_generate_gif_from_frames(landscape_image, temp_output_dir):
    """GIF generated from frame sequence."""
    from imageloop.job import generate_gif
    
    images_dir = temp_output_dir / "images"
    images_dir.mkdir()
    
    # Create 3 test frames
    for i in range(3):
        frame_path = images_dir / f"frame_{i:03d}.png"
        img = Image.open(landscape_image)
        img.save(frame_path)
        img.close()
    
    gif_path = temp_output_dir / "test.gif"
    success = generate_gif(images_dir, gif_path, frame_rate=1)
    
    assert success is True
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0


def test_generate_gif_fails_with_insufficient_frames(temp_output_dir):
    """GIF generation fails if less than 2 frames."""
    from imageloop.job import generate_gif
    
    images_dir = temp_output_dir / "images"
    images_dir.mkdir()
    
    # Create only 1 frame
    frame_path = images_dir / "frame_001.png"
    img = Image.new("RGB", (100, 100))
    img.save(frame_path)
    img.close()
    
    gif_path = temp_output_dir / "test.gif"
    success = generate_gif(images_dir, gif_path)
    
    assert success is False
    assert not gif_path.exists()


def test_generate_outputs_creates_gif(landscape_image, temp_output_dir):
    """generate_outputs creates GIF when format is 'gif'."""
    from imageloop.job import generate_outputs
    
    images_dir = temp_output_dir / "images"
    images_dir.mkdir()
    
    # Create 3 test frames
    for i in range(3):
        frame_path = images_dir / f"frame_{i:03d}.png"
        img = Image.open(landscape_image)
        img.save(frame_path)
        img.close()
    
    run_dir = temp_output_dir / "run"
    run_dir.mkdir()
    
    generate_outputs(images_dir, run_dir, frame_rate=1, output_format="gif")
    
    gif_path = run_dir / "animation.gif"
    assert gif_path.exists()


def test_generate_outputs_creates_both(landscape_image, temp_output_dir):
    """generate_outputs creates both MP4 and GIF when format is 'both'."""
    from imageloop.job import generate_outputs
    import subprocess
    
    # Check if ffmpeg is available (both ffmpeg binary and python package)
    has_ffmpeg_bin = False
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        has_ffmpeg_bin = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    has_ffmpeg_python = False
    try:
        import ffmpeg
        has_ffmpeg_python = True
    except ImportError:
        pass
    
    if not (has_ffmpeg_bin and has_ffmpeg_python):
        pytest.skip("ffmpeg not available for MP4 generation test")
    
    images_dir = temp_output_dir / "images"
    images_dir.mkdir()
    
    # Create 3 test frames
    for i in range(3):
        frame_path = images_dir / f"frame_{i:03d}.png"
        img = Image.open(landscape_image)
        img.save(frame_path)
        img.close()
    
    run_dir = temp_output_dir / "run"
    run_dir.mkdir()
    
    generate_outputs(images_dir, run_dir, frame_rate=1, output_format="both")
    
    gif_path = run_dir / "animation.gif"
    mp4_path = run_dir / "animation.mp4"
    
    assert gif_path.exists()
    assert mp4_path.exists()
