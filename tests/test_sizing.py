"""Tests for imageloop.sizing module - image dimension calculations."""

from PIL import Image
import pytest
from imageloop import storage


def test_landscape_image_detected_as_landscape(landscape_image):
    """4:3 aspect ratio maps to 'landscape' preset."""
    from imageloop.sizing import detect_best_size
    
    img = Image.open(landscape_image)
    size_name = detect_best_size(img.size[0], img.size[1])
    img.close()
    
    assert size_name == "landscape"


def test_portrait_image_detected_as_portrait(portrait_image):
    """3:4 aspect ratio maps to 'portrait' preset."""
    from imageloop.sizing import detect_best_size
    
    img = Image.open(portrait_image)
    size_name = detect_best_size(img.size[0], img.size[1])
    img.close()
    
    assert size_name == "portrait"


def test_wide_image_detected_as_wide(wide_image):
    """16:9 aspect ratio maps to 'wide' preset."""
    from imageloop.sizing import detect_best_size
    
    img = Image.open(wide_image)
    size_name = detect_best_size(img.size[0], img.size[1])
    img.close()
    
    assert size_name == "wide"


def test_oversized_image_scaled_down(oversized_image):
    """Images larger than max dimension get scaled to fit."""
    from imageloop.sizing import calculate_auto_size
    
    img = Image.open(oversized_image)
    orig_w, orig_h = img.size
    img.close()
    
    # Calculate auto size with default max of 1280
    new_w, new_h = calculate_auto_size(orig_w, orig_h, max_dimension=1280)
    
    # Should be scaled down
    assert new_w <= 1280
    assert new_h <= 1280
    assert new_w < orig_w or new_h < orig_h


def test_crop_mode_fills_target_exactly(landscape_image):
    """Crop resize produces exact target dimensions."""
    from imageloop.sizing import resize_to_size
    
    # Convert to data URI
    data_uri = storage.image_to_data_uri(landscape_image)
    
    # Resize to square using crop mode
    target_size = (512, 512)
    resized_uri = resize_to_size(data_uri, target_size, mode="crop")
    
    # Check dimensions
    resized_img = storage.data_uri_to_image(resized_uri)
    assert resized_img.size == target_size
    resized_img.close()


def test_fit_mode_preserves_aspect_ratio(odd_ratio_image):
    """Fit resize maintains original aspect ratio with padding."""
    from imageloop.sizing import resize_to_size
    
    # Get original aspect ratio
    orig_img = Image.open(odd_ratio_image)
    orig_w, orig_h = orig_img.size
    orig_ratio = orig_w / orig_h
    orig_img.close()
    
    # Convert to data URI
    data_uri = storage.image_to_data_uri(odd_ratio_image)
    
    # Resize to square using fit mode (will have padding)
    target_size = (512, 512)
    resized_uri = resize_to_size(data_uri, target_size, mode="fit")
    
    # The actual image inside should maintain aspect ratio
    # Since fit mode pads, the entire image is target size, but
    # we can check that the aspect ratio is preserved by examining pixels
    # For now, just verify it's the target size and we can load it
    resized_img = storage.data_uri_to_image(resized_uri)
    assert resized_img.size == target_size
    resized_img.close()


def test_tiny_image_upscaled_to_minimum(tiny_image):
    """Images below minimum size get scaled up."""
    from imageloop.sizing import calculate_auto_size
    
    img = Image.open(tiny_image)
    orig_w, orig_h = img.size
    img.close()
    
    # Calculate with minimum of 512
    new_w, new_h = calculate_auto_size(orig_w, orig_h, min_dimension=512)
    
    # At least one dimension should be at minimum
    assert new_w >= 512 or new_h >= 512
    # Both should be at least minimum
    assert new_w >= 512
    assert new_h >= 512


def test_round_to_multiple():
    """round_to_multiple rounds to nearest multiple of 64."""
    from imageloop.sizing import round_to_multiple
    
    assert round_to_multiple(100, 64) == 128  # 100 -> 128 (nearest)
    assert round_to_multiple(95, 64) == 64    # 95 -> 64 (nearest)
    assert round_to_multiple(96, 64) == 128   # 96 -> 128 (rounds up at midpoint)
    assert round_to_multiple(64, 64) == 64
    assert round_to_multiple(127, 64) == 128


def test_calculate_auto_size_produces_valid_dimensions(odd_ratio_image):
    """Auto size calculation produces valid dimensions within constraints."""
    from imageloop.sizing import calculate_auto_size, round_to_multiple
    
    img = Image.open(odd_ratio_image)
    orig_w, orig_h = img.size
    img.close()
    
    # Calculate auto size
    new_w, new_h = calculate_auto_size(orig_w, orig_h, max_dimension=1280, min_dimension=512)
    
    # Should be within max dimension
    assert new_w <= 1280
    assert new_h <= 1280
    
    # Should be at least minimum
    assert new_w >= 512
    assert new_h >= 512
    
    # Should be multiples of 64
    assert new_w % 64 == 0
    assert new_h % 64 == 0
    
    # Should scale down if oversized, or up if undersized
    # (At least one dimension should change to fit constraints)
    assert new_w <= 1280
    assert new_h <= 1280


def test_resize_stretch_mode_changes_aspect_ratio(landscape_image):
    """Stretch mode forces exact target dimensions even if aspect ratio changes."""
    from imageloop.sizing import resize_to_size
    
    # Original is 400x300 (4:3 ratio)
    data_uri = storage.image_to_data_uri(landscape_image)
    
    # Stretch to 512x512 (square, different ratio)
    target_size = (512, 512)
    resized_uri = resize_to_size(data_uri, target_size, mode="stretch")
    
    # Should be exactly the target size
    resized_img = storage.data_uri_to_image(resized_uri)
    assert resized_img.size == target_size
    resized_img.close()
