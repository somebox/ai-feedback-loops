"""Image dimension calculations and resize operations."""

from imageloop import storage

# Standard sizes for image generation (width, height)
STANDARD_SIZES = {
    "square": (1024, 1024),      # 1:1
    "landscape": (1024, 768),    # 4:3
    "portrait": (768, 1024),     # 3:4
    "wide": (1280, 720),         # 16:9
    "tall": (720, 1280),         # 9:16
}

# Special size modes (not in STANDARD_SIZES)
SPECIAL_SIZES = ["auto", "preserve", "custom"]

DEFAULT_SIZE = "auto"


def detect_best_size(width: int, height: int) -> str:
    """
    Detect the best standard size based on input image dimensions.
    Returns the size name that best matches the aspect ratio.
    """
    input_ratio = width / height
    
    best_size = None
    best_diff = float("inf")
    
    for name, (w, h) in STANDARD_SIZES.items():
        target_ratio = w / h
        diff = abs(input_ratio - target_ratio)
        if diff < best_diff:
            best_diff = diff
            best_size = name
    
    return best_size


def round_to_multiple(value: int, multiple: int = 64) -> int:
    """Round a value to the nearest multiple (for GPU-friendly dimensions)."""
    return ((value + multiple // 2) // multiple) * multiple


def calculate_auto_size(width: int, height: int, max_dimension: int = 1280, min_dimension: int = 512) -> tuple[int, int]:
    """
    Calculate output dimensions that preserve aspect ratio.
    
    Scales to fit within max_dimension while maintaining aspect ratio,
    and rounds to multiples of 64 for GPU efficiency.
    """
    # Scale down if needed
    scale = min(max_dimension / max(width, height), 1.0)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Round to multiples of 64
    new_width = round_to_multiple(new_width, 64)
    new_height = round_to_multiple(new_height, 64)
    
    # Ensure minimum size
    new_width = max(new_width, min_dimension)
    new_height = max(new_height, min_dimension)
    
    return new_width, new_height


def resize_to_size(data_uri: str, target_size: tuple[int, int], verbose: bool = False, mode: str = "stretch") -> str:
    """
    Resize image to exact dimensions.
    
    Args:
        data_uri: Input image as data URI
        target_size: Tuple of (width, height)
        verbose: Log size changes
        mode: "stretch" (stretch to fit, may distort), "fit" (letterbox/pad), "crop" (center crop)
    
    Returns:
        Resized image as data URI
    """
    from PIL import Image
    import io
    import base64
    
    target_w, target_h = target_size
    image = storage.data_uri_to_image(data_uri)
    
    # Convert to RGB if needed
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    
    orig_w, orig_h = image.size
    
    # Skip if already correct size
    if orig_w == target_w and orig_h == target_h:
        return data_uri
    
    if verbose:
        print(f"    📏 Resizing ({mode}): {orig_w}x{orig_h} → {target_w}x{target_h}")
    
    if mode == "stretch":
        # Simply stretch to target size (may cause minor distortion)
        image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    elif mode == "fit":
        # Scale to fit within target, then pad (letterbox)
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create new image with black background and paste centered
        result = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        result.paste(image, (paste_x, paste_y))
        image = result
    else:
        # mode == "crop": Scale to cover target, then center crop
        scale = max(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Center crop to target size
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        image = image.crop((left, top, left + target_w, top + target_h))
    
    # Encode back to data URI
    output_buffer = io.BytesIO()
    image.save(output_buffer, format="PNG", optimize=False)
    encoded_string = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
    
    return f"data:image/png;base64,{encoded_string}"


def standardize_image(data_uri: str, size: str = DEFAULT_SIZE, custom_size: tuple[int, int] = None) -> tuple[str, tuple[int, int], str]:
    """
    Resize and crop initial image to a standard size for consistent video frames.
    
    Args:
        data_uri: Input image as data URI
        size: Size preset name ("auto", "preserve", "square", "landscape", etc.) or "custom"
        custom_size: Tuple of (width, height) when size is "custom"
    
    Returns:
        Tuple of (data_uri, (width, height), size_name)
    """
    image = storage.data_uri_to_image(data_uri)
    orig_w, orig_h = image.size
    image.close()
    
    # Handle special size modes
    if size == "auto":
        # Pick best matching standard size
        size = detect_best_size(orig_w, orig_h)
        target_size = STANDARD_SIZES[size]
    elif size == "preserve":
        # Preserve aspect ratio, scale to reasonable dimensions
        target_size = calculate_auto_size(orig_w, orig_h)
        size = f"preserve ({target_size[0]}x{target_size[1]})"
    elif size == "custom":
        if not custom_size:
            raise ValueError("Custom size requires --width and --height")
        target_size = custom_size
        size = f"custom ({target_size[0]}x{target_size[1]})"
    elif size in STANDARD_SIZES:
        target_size = STANDARD_SIZES[size]
    else:
        raise ValueError(f"Unknown size: {size}. Available: auto, preserve, custom, {', '.join(STANDARD_SIZES.keys())}")
    
    # Use crop mode for initial image to fill the frame
    resized = resize_to_size(data_uri, target_size, mode="crop")
    return resized, target_size, size
