"""Settings loading and configuration management."""

import yaml
from pathlib import Path
from typing import Dict, Any


# Embedded defaults (used when settings.yaml doesn't exist)
DEFAULT_SETTINGS = {
    "models": {
        "flux-pro": "black-forest-labs/flux.2-pro",
        "flux-klein": "black-forest-labs/flux.2-klein-4b",
        "seedream": "bytedance-seed/seedream-4.5",
        "nano-banana": "google/gemini-2.5-flash-image",
        "nano-banana-pro": "google/gemini-3-pro-image-preview",
        "riverflow": "sourceful/riverflow-v2-standard-preview",
    },
    "describe_prompts": {
        "detailed": "Describe this image in detail, including the subject matter, composition, colors, lighting, mood, and any notable details.",
        "artistic": "Describe this image as an art director would, focusing on visual style, composition, color palette, lighting techniques, and artistic elements.",
        "simple": "Briefly describe what you see in this image.",
        "technical": "Provide a technical description of this image including dimensions, aspect ratio, color distribution, focal points, and visual hierarchy.",
        "narrative": "Describe this image as if telling a story. What is happening? What might have happened before or after this moment?",
        "emotional": "Describe the emotional quality and atmosphere of this image. What feelings does it evoke and why?",
    },
    "prompts": {
        "up": "Gently pan the camera up, extending the image.",
        "down": "Gently pan the camera down, extending the image.",
        "left": "Gently pan the camera left, extending the image.",
        "right": "Gently pan the camera right, extending the image.",
        "orbit": "The camera is on a circular track orbiting the subject; move the camera along that track to the left a bit and show what the image looks like from that new perspective",
        "rotate": "Gently rotate the camera clockwise, extending the borders to fit the new perspective.",
        "zoom-in": "Gently zoom in on the center of the image, maintaining focus and detail.",
        "zoom-out": "Gently zoom out from the image, revealing more of the surrounding scene.",
        "future": "Show this scene one second in the future",
        "past": "Show this scene one second in the past",
        "funny": "Subtly alter this image by replacing one or two details with something that makes the image more humorous or silly.",
        "highlight": "Subtly alter this image to bring more attention to a subtle detail",
        "dramatic": "Subtly enhance the drama and intensity of this scene. Adjust lighting to be more cinematic, deepen shadows, or add atmospheric elements like mist or dramatic sky.",
        "peaceful": "Transform this scene to be more peaceful and serene. Soften harsh elements, add calming details like gentle lighting or natural elements.",
        "powerful": "Transform this scene to be more powerful and intense. Make it slightly more intense and extreme.",
        "vintage": "Apply a subtle vintage aesthetic to this image. Add slight film grain, adjust colors to warmer or cooler vintage tones, and create a nostalgic atmosphere.",
        "futuristic": "Subtly modernize or add futuristic elements to this scene. Replace one or two objects with sleek, high-tech alternatives.",
        "nature": "Subtly introduce natural elements into this scene. Add plants, natural lighting, or organic textures.",
        "urban": "Subtly add urban elements to this scene. Introduce architectural details, city textures, or modern infrastructure.",
        "minimalist": "Simplify this scene with minimalist aesthetics. Remove or tone down distracting elements, create cleaner compositions.",
        "bizarre": "Subtly alter this image by replacing one or two details with something slightly unexpected and bizarre.",
        "wes-anderson": "Adjust this image so it look a bit more like a Wes Anderson movie.",
        "corrections": "Find something wrong with this image and fix it.",
        "crowded": "Subtly add more people or objects to make this scene feel more populated or busy.",
        "empty": "Subtly remove people or objects to make this scene feel more spacious or isolated.",
        "evolve": "Transform this image slightly, letting it evolve naturally in an interesting direction.",
        "cooler": "make this image and any people in it more 'cool' (style, not temperature)",
        "sexy": "make this image seem more 'sexy' and alluring",
        "politic-right": "how would this image look if it was just a bit more 'politically right' or conservative",
        "politic-left": "how would this image look if it was just a bit more 'politically left' or liberal",
        "makeup": "make this image more glamorous with extra makeup, eyeliner, fancier hair, etc.",
        "album-cover": "modify this image slightly so that it looks more like an album cover",
        "graffiti": "add some graffiti to this image, making it look more urban and edgy",
        "realistic": "make this image more realistic, fixing any fake or unrealistic elements",
        "next": "show what happens moments later in this scene",
        "opposite": "consider the deeper meaning of this image and show the opposite of what is shown",
        "improve": "review this image and improve it with optimizations, corrections, or design improvements",
        "unexpected": "show something unexpected that happened just a second after this image was taken",
        "redacted": "replace any redacted black square with the likely original content that was obscured, then find a new area of the image to redact and cover it with a black square.",
        "hidden": "find something that is hidden from view in this image, and remove the item obstructing the view",
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


def load_settings(settings_path: Path = None) -> Dict[str, Any]:
    """
    Load settings from YAML file, falling back to embedded defaults.
    
    Args:
        settings_path: Path to settings.yaml file. If None, looks for settings.yaml
                      in project root. If file doesn't exist, uses embedded defaults.
    
    Returns:
        Dictionary with settings (models, prompts, defaults, sizes, api)
    """
    if settings_path is None:
        # Look for settings.yaml in project root (two levels up from this file)
        settings_path = Path(__file__).parent.parent.parent / "settings.yaml"
    
    if settings_path.exists():
        with open(settings_path) as f:
            user_settings = yaml.safe_load(f) or {}
        
        # Merge with defaults (user settings override defaults)
        # Need deep copy to avoid mutating defaults
        import copy
        settings = copy.deepcopy(DEFAULT_SETTINGS)
        for key, value in user_settings.items():
            if isinstance(value, dict) and key in settings:
                settings[key].update(value)
            else:
                settings[key] = value
    else:
        # Use embedded defaults (return copy to avoid mutation)
        import copy
        settings = copy.deepcopy(DEFAULT_SETTINGS)
    
    return settings


def get_model(model_name: str, settings: Dict[str, Any]) -> str:
    """
    Resolve model name to full model ID.
    
    Args:
        model_name: Model shortcut (e.g., "flux-pro") or full model ID
        settings: Settings dictionary
    
    Returns:
        Full model ID (passes through if already a full ID)
    """
    models = settings.get("models", {})
    
    # Check if it's a shortcut
    if model_name in models:
        return models[model_name]
    
    # Assume it's already a full model ID
    return model_name


def get_prompt(mode: str, custom_prompt: str = None, settings: Dict[str, Any] = None) -> str:
    """
    Get prompt for a given mode or custom prompt.
    
    Args:
        mode: Mode name (preset) or "custom"
        custom_prompt: Custom prompt text (required when mode is "custom")
        settings: Settings dictionary (if None, loads from default location)
    
    Returns:
        Prompt text
    
    Raises:
        ValueError: If mode is unknown or custom mode lacks prompt
    """
    if settings is None:
        settings = load_settings()
    
    prompts = settings.get("prompts", {})
    
    if mode == "custom":
        if not custom_prompt:
            raise ValueError("Custom prompt required when mode is 'custom'")
        return custom_prompt
    
    if mode in prompts:
        return prompts[mode]
    
    available = sorted(prompts.keys())
    raise ValueError(
        f"Unknown mode: {mode}. Available: {', '.join(available)}, or 'custom' with --prompt"
    )


def get_describe_prompt(mode: str, custom_prompt: str = None, settings: Dict[str, Any] = None) -> str:
    """
    Get describe prompt for prompt-loop mode.
    
    Args:
        mode: Describe mode name (preset) or "custom"
        custom_prompt: Custom prompt text (required when mode is "custom")
        settings: Settings dictionary (if None, loads from default location)
    
    Returns:
        Describe prompt text
    
    Raises:
        ValueError: If mode is unknown or custom mode lacks prompt
    """
    if settings is None:
        settings = load_settings()
    
    describe_prompts = settings.get("describe_prompts", {})
    
    if mode == "custom":
        if not custom_prompt:
            raise ValueError("Custom describe prompt required when describe mode is 'custom'")
        return custom_prompt
    
    if mode in describe_prompts:
        return describe_prompts[mode]
    
    available = sorted(describe_prompts.keys())
    raise ValueError(
        f"Unknown describe mode: {mode}. Available: {', '.join(available)}, or 'custom' with --describe-prompt"
    )


def get_model_short_name(model_id: str, settings: Dict[str, Any] = None) -> str:
    """
    Get the short name for a model, or create one from the full ID.
    
    Args:
        model_id: Full model ID (e.g., "black-forest-labs/flux.2-pro")
        settings: Settings dictionary (if None, loads from default location)
    
    Returns:
        Short name (e.g., "flux-pro") or sanitized model ID if not found
    """
    if settings is None:
        settings = load_settings()
    
    # Build reverse lookup
    models = settings.get("models", {})
    models_reverse = {v: k for k, v in models.items()}
    
    if model_id in models_reverse:
        return models_reverse[model_id]
    
    # Fall back to sanitizing the full model ID
    return model_id.replace("/", "-").replace(".", "-")[:20]
