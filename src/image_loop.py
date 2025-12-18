# /// script
# dependencies = [
#   "httpx",
#   "Pillow",
#   "ffmpeg-python",
#   "tqdm",
#   "pyyaml",
# ]
# ///

"""
Image Loop Generator - Iterative AI image generation using OpenRouter

Run with: uv run src/image_loop.py --help
"""

import argparse
import asyncio
import base64
import io
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import ffmpeg
import httpx
import yaml
from PIL import Image
from tqdm import tqdm

# Available models for image generation
MODELS = {
    "flux-pro": "black-forest-labs/flux.2-pro",
    # "gemini-image": "google/gemini-3-pro-image-preview",
    "gemini-flash-image": "google/gemini-2.5-flash-image",
    "riverflow": "sourceful/riverflow-v2-standard-preview",
}

# Preset prompts for common operations
PROMPTS = {
    "up": "Gently pan the camera up, extending the image.",
    "down": "Gently pan the camera down, extending the image.",
    "left": "Gently pan the camera left, extending the image.",
    "right": "Gently pan the camera right, extending the image.",
    "rotate-left": "Gently rotate the camera counter-clockwise, extending the borders to fit the new perspective.",
    "rotate-right": "Gently rotate the camera clockwise, extending the borders to fit the new perspective.",
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
    "improve": "review this image and improve it with optimizations, corrections, or design improvements"
}

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/responses"


class GenerationStats:
    """Track statistics for the generation run."""

    def __init__(self):
        self.start_time = time.time()
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_cost = 0.0
        self.frames_generated = 0
        self.frames_failed = 0
        self.api_calls = 0

    def add_response(self, usage: dict):
        """Add token usage from a response."""
        self.api_calls += 1
        if usage:
            # OpenRouter uses input_tokens/output_tokens
            self.input_tokens += usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
            # OpenRouter includes cost in usage
            self.total_cost += usage.get("cost", 0.0)

    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def report(self) -> str:
        """Generate a summary report."""
        elapsed = self.elapsed_time()
        lines = [
            "",
            "=" * 50,
            "📊 Generation Report",
            "=" * 50,
            f"⏱️  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)",
            f"🖼️  Frames generated: {self.frames_generated}",
            f"❌ Frames failed: {self.frames_failed}",
            f"🔄 API calls: {self.api_calls}",
            f"📥 Input tokens: {self.input_tokens:,}",
            f"📤 Output tokens: {self.output_tokens:,}",
            f"📊 Total tokens: {self.total_tokens:,}",
            f"💰 Total cost: ${self.total_cost:.4f}",
            "=" * 50,
        ]
        if self.frames_generated > 0:
            lines.insert(-1, f"⚡ Avg time per frame: {elapsed/self.frames_generated:.1f}s")
            lines.insert(-1, f"💵 Cost per frame: ${self.total_cost/self.frames_generated:.4f}")
        return "\n".join(lines)


def load_api_key() -> str:
    """Load OpenRouter API key from environment or secrets.yaml."""
    # Check environment first
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return api_key

    # Try secrets.yaml in project root
    secrets_path = Path(__file__).parent.parent / "secrets.yaml"
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


# Standard sizes for image generation (width, height)
STANDARD_SIZES = {
    "square": (1024, 1024),
    "landscape": (1024, 768),    # 4:3
    "portrait": (768, 1024),     # 3:4
    "wide": (1280, 720),         # 16:9
    "tall": (720, 1280),         # 9:16
}

DEFAULT_SIZE = "landscape"


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
    target_w, target_h = target_size
    image = data_uri_to_image(data_uri)
    
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


def standardize_image(data_uri: str, size: str = DEFAULT_SIZE) -> tuple[str, tuple[int, int]]:
    """
    Resize and crop initial image to a standard size for consistent video frames.
    
    Uses center crop to fill the target dimensions (crops edges if needed).
    
    Returns:
        Tuple of (data_uri, (width, height))
    """
    if size not in STANDARD_SIZES:
        raise ValueError(f"Unknown size: {size}. Available: {list(STANDARD_SIZES.keys())}")
    
    target_size = STANDARD_SIZES[size]
    # Use crop mode for initial image to fill the frame
    resized = resize_to_size(data_uri, target_size, mode="crop")
    return resized, target_size


async def fetch_image_url(url: str) -> str:
    """Fetch an image from URL and return as data URI."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "image/png")
        encoded = base64.b64encode(response.content).decode("utf-8")
        return f"data:{content_type};base64,{encoded}"


async def generate_image(
    prompt: str,
    image_data_uri: str,
    model: str,
    api_key: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stats: GenerationStats = None,
    verbose: bool = False,
) -> tuple[str | None, dict]:
    """
    Generate a new image using the OpenRouter responses API.

    Returns:
        Tuple of (image_data_uri or None, usage_dict)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ai-feedback-loops",
        "X-Title": "Image Loop Generator",
    }

    # Build the request payload
    payload = {
        "model": model,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": image_data_uri,
                        "detail": "high",
                    },
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                ],
            }
        ],
        "temperature": temperature,
        "top_p": top_p,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
            )

            if response.status_code != 200:
                error_text = response.text
                print(f"\n❌ API Error ({response.status_code}): {error_text[:200]}")
                return None, {}

            result = response.json()

            if verbose:
                import json
                print(f"\n📡 API Response keys: {list(result.keys())}")
                if result.get("usage"):
                    print(f"📊 Usage: {json.dumps(result['usage'], indent=2)}")
                if result.get("output"):
                    print(f"📤 Output types: {[item.get('type') for item in result['output']]}")

            # Extract usage info
            usage = result.get("usage", {})
            if stats:
                stats.add_response(usage)

            # Extract the generated image from the response
            output = result.get("output", [])
            for item in output:
                item_type = item.get("type")
                
                # Handle message type output
                if item_type == "message":
                    content = item.get("content", [])
                    for part in content:
                        part_type = part.get("type")
                        # Check for image output
                        if part_type in ("output_image", "image"):
                            image_url = part.get("image_url") or part.get("url")
                            if image_url:
                                if image_url.startswith("http"):
                                    return await fetch_image_url(image_url), usage
                                return image_url, usage

                        # Some models return base64 directly
                        if part_type == "image" and part.get("data"):
                            mime = part.get("mime_type", "image/png")
                            return f"data:{mime};base64,{part['data']}", usage

                        # Check for image_generation_call type
                        if part_type == "image_generation_call" and part.get("result"):
                            img_result = part.get("result")
                            if isinstance(img_result, str):
                                if img_result.startswith("http"):
                                    return await fetch_image_url(img_result), usage
                                return img_result, usage

                # Handle image_generation_call at output level
                if item_type == "image_generation_call":
                    img_result = item.get("result")
                    if img_result:
                        if isinstance(img_result, str):
                            if img_result.startswith("http"):
                                return await fetch_image_url(img_result), usage
                            return img_result, usage

            # Check for direct image in result
            if "image" in result:
                img = result["image"]
                if isinstance(img, str):
                    if img.startswith("http"):
                        return await fetch_image_url(img), usage
                    elif img.startswith("data:"):
                        return img, usage

            # Check output_text for a URL (some models return image URL in text)
            output_text = result.get("output_text", "")
            if output_text:
                # Look for URLs in the text
                import re
                url_pattern = r'https?://[^\s<>"\']+\.(?:png|jpg|jpeg|gif|webp)[^\s<>"\']*'
                urls = re.findall(url_pattern, output_text, re.IGNORECASE)
                if urls:
                    return await fetch_image_url(urls[0]), usage

            # Check for error in response
            error = result.get("error")
            if error:
                error_msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
                print(f"\n⚠️  Model error: {error_msg[:150]}")
                return None, usage

            # Extract text from message output (often contains refusal reason)
            model_text = output_text
            if not model_text:
                for item in output:
                    if item.get("type") == "message":
                        content = item.get("content", [])
                        for part in content:
                            if part.get("type") == "output_text":
                                model_text = part.get("text", "")
                                break
                            elif part.get("type") == "text":
                                model_text = part.get("text", "")
                                break
                        if model_text:
                            break

            # No image found - show what the model said (often a content policy refusal)
            print(f"\n⚠️  No image generated.")
            if model_text:
                # Truncate long responses
                text_preview = model_text[:300] + "..." if len(model_text) > 300 else model_text
                print(f"    Model said: {text_preview}")
            else:
                print(f"    Output types: {[item.get('type') for item in output]}")
            return None, usage

        except httpx.TimeoutException:
            print("\n❌ Request timed out")
            return None, {}
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return None, {}


def generate_video(images_dir: Path, output_file: Path, frame_rate: int = 12):
    """Generate an MP4 video from a directory of frame images."""
    import glob

    frame_pattern = str(images_dir / "frame_*.png")
    frame_files = sorted(glob.glob(frame_pattern))
    num_frames = len(frame_files)

    if num_frames < 2:
        print(f"⚠️  Not enough frames ({num_frames}) to generate video")
        return False

    print(f"\n🎥 Generating video from {num_frames} frames...")

    try:
        (
            ffmpeg.input(
                frame_pattern,
                pattern_type="glob",
                framerate=frame_rate,
            )
            .output(str(output_file), vcodec="libx264", pix_fmt="yuv420p")
            .run(overwrite_output=True, quiet=True)
        )

        file_size = output_file.stat().st_size / (1024 * 1024)
        print(f"✅ Video saved: {output_file} ({file_size:.1f} MB)")
        return True

    except ffmpeg.Error as e:
        stderr = e.stderr.decode("utf8") if e.stderr else str(e)
        print(f"❌ Video generation failed: {stderr}")
        return False


def get_prompt(mode: str, custom_prompt: str = None) -> str:
    """Get the prompt for a given mode or custom prompt."""
    if mode == "custom":
        if not custom_prompt:
            raise ValueError("Custom prompt required when mode is 'custom'")
        return custom_prompt

    if mode in PROMPTS:
        return PROMPTS[mode]

    raise ValueError(f"Unknown mode: {mode}. Available: {list(PROMPTS.keys())}")


def find_last_frame(images_dir: Path) -> tuple[int, Path] | None:
    """Find the last frame in a directory. Returns (frame_number, path) or None."""
    import glob
    frame_files = glob.glob(str(images_dir / "frame_*.png"))
    if not frame_files:
        return None
    
    # Sort by frame number
    def get_frame_num(path):
        name = Path(path).stem
        try:
            return int(name.split("_")[1])
        except (IndexError, ValueError):
            return -1
    
    frame_files.sort(key=get_frame_num)
    last_frame = Path(frame_files[-1])
    last_num = get_frame_num(str(last_frame))
    return last_num, last_frame


def parse_report(report_path: Path) -> dict:
    """Parse settings from an existing report.txt file."""
    settings = {}
    if not report_path.exists():
        return settings
    
    with open(report_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Model: "):
                settings["model"] = line[7:]
            elif line.startswith("Mode: "):
                settings["mode"] = line[6:]
            elif line.startswith("Prompt: "):
                settings["prompt"] = line[8:]
            elif line.startswith("Temperature: "):
                settings["temperature"] = float(line[13:])
            elif line.startswith("Top P: "):
                settings["top_p"] = float(line[7:])
            elif line.startswith("Size: "):
                # Parse "Size: square (1024x1024)" -> "square"
                size_part = line[6:].split(" ")[0]
                settings["size"] = size_part
    
    return settings


async def run_continue(args):
    """Continue an existing generation run."""
    # Load API key
    try:
        api_key = load_api_key()
    except ValueError as e:
        print(f"❌ {e}")
        return 1

    # Find the run directory
    run_dir = Path(args.continue_run)
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        return 1

    # Handle if user passed the images subdirectory
    if run_dir.name == "images":
        run_dir = run_dir.parent

    images_dir = run_dir / "images"
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return 1

    # Find last frame
    result = find_last_frame(images_dir)
    if result is None:
        print(f"❌ No frames found in {images_dir}")
        return 1
    
    last_frame_num, last_frame_path = result
    print(f"📁 Continuing from: {run_dir}")
    print(f"🖼️  Last frame: {last_frame_path.name} (frame {last_frame_num})")

    # Load settings from report if available
    report_path = run_dir / "report.txt"
    saved_settings = parse_report(report_path)
    
    # Model priority: explicit CLI arg > saved from report > default
    if args.model:
        model = MODELS.get(args.model, args.model)
    elif saved_settings.get("model"):
        model = saved_settings["model"]
    else:
        model = MODELS.get("flux-pro", "flux-pro")
    
    # Get prompt - prefer CLI args, fall back to saved
    if args.mode:
        prompt = get_prompt(args.mode, args.prompt)
    elif saved_settings.get("prompt"):
        prompt = saved_settings["prompt"]
    else:
        print("❌ No mode specified and couldn't find prompt in report")
        return 1

    temperature = args.temperature if args.temperature != 0.7 else saved_settings.get("temperature", 0.7)
    top_p = args.top_p if args.top_p != 0.9 else saved_settings.get("top_p", 0.9)

    # Get frame size from saved settings or detect from last frame
    if saved_settings.get("size") and saved_settings["size"] in STANDARD_SIZES:
        frame_size = STANDARD_SIZES[saved_settings["size"]]
    else:
        # Detect size from last frame
        last_image = Image.open(last_frame_path)
        frame_size = last_image.size
        last_image.close()
    
    print(f"🤖 Model: {model}")
    print(f"💬 Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"📐 Frame size: {frame_size[0]}x{frame_size[1]}")

    # Load the last frame as current image
    print(f"📷 Loading last frame...")
    current_image = image_to_data_uri(last_frame_path)

    # Initialize stats
    stats = GenerationStats()

    # Generation loop
    print(f"\n🎬 Generating {args.frames} more frames...\n")

    progress = tqdm(range(args.frames), desc="Generating", unit="frame")

    for i in progress:
        frame_num = last_frame_num + i + 1

        new_image, usage = await generate_image(
            prompt=prompt,
            image_data_uri=current_image,
            model=model,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            stats=stats,
            verbose=args.verbose,
        )

        if new_image is None:
            stats.frames_failed += 1
            progress.set_postfix({"status": "failed", "total": stats.frames_generated})
            continue

        # Resize to match existing frame size
        new_image = resize_to_size(new_image, frame_size, verbose=args.verbose)

        # Save frame
        frame_path = images_dir / f"frame_{frame_num:03d}"
        try:
            file_size = save_data_uri(new_image, frame_path)
            stats.frames_generated += 1
            current_image = new_image

            progress.set_postfix({
                "size": f"{file_size // 1024}KB",
                "tokens": stats.total_tokens,
            })

        except Exception as e:
            print(f"\n❌ Failed to save frame {frame_num}: {e}")
            stats.frames_failed += 1

    # Regenerate video with all frames
    if stats.frames_generated > 0:
        video_path = run_dir / "animation.mp4"
        generate_video(images_dir, video_path, args.fps)

    # Print report
    print(stats.report())

    # Append to report file
    report_path = run_dir / "report.txt"
    with open(report_path, "a") as f:
        f.write(f"\n\n--- Continued at {datetime.now().isoformat()} ---\n")
        f.write(f"Added frames: {args.frames}\n")
        f.write(f"Starting from frame: {last_frame_num}\n")
        f.write(stats.report())

    print(f"\n📝 Report updated: {report_path}")

    # Show total frame count
    total_result = find_last_frame(images_dir)
    if total_result:
        print(f"📊 Total frames now: {total_result[0] + 1}")

    return 0 if stats.frames_generated > 0 else 1


async def run_generation(args):
    """Main generation loop."""
    # Load API key
    try:
        api_key = load_api_key()
    except ValueError as e:
        print(f"❌ {e}")
        return 1

    # Resolve model name (default to flux-pro if not specified)
    model_arg = args.model or "flux-pro"
    model = MODELS.get(model_arg, model_arg)
    print(f"🤖 Model: {model}")

    # Load input image
    input_path = Path(args.image)
    if not input_path.exists():
        print(f"❌ Image not found: {input_path}")
        return 1

    print(f"📷 Loading image: {input_path}")
    current_image = image_to_data_uri(input_path)

    # Standardize to target size for consistent frames
    current_image, frame_size = standardize_image(current_image, args.size)
    print(f"📐 Standardized to {frame_size[0]}x{frame_size[1]} ({args.size})")

    # Get prompt
    try:
        prompt = get_prompt(args.mode, args.prompt)
    except ValueError as e:
        print(f"❌ {e}")
        return 1

    print(f"💬 Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    # Set up output directory
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_id = uuid.uuid4().hex[:4]
    model_short = model.replace("/", "-").replace(".", "-")[:20]
    mode_short = args.mode[:15]

    output_base = Path(args.output)
    run_dir = output_base / f"run_{model_short}_{mode_short}_{timestamp}_{run_id}"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output: {run_dir}")

    # Save initial frame
    save_data_uri(current_image, images_dir / "frame_000")
    print("✅ Saved initial frame (frame_000.png)")

    # Initialize stats
    stats = GenerationStats()

    # Generation loop
    print(f"\n🎬 Generating {args.frames} frames...\n")

    progress = tqdm(range(args.frames), desc="Generating", unit="frame")

    for i in progress:
        frame_num = i + 1

        new_image, usage = await generate_image(
            prompt=prompt,
            image_data_uri=current_image,
            model=model,
            api_key=api_key,
            temperature=args.temperature,
            top_p=args.top_p,
            stats=stats,
            verbose=args.verbose,
        )

        if new_image is None:
            stats.frames_failed += 1
            progress.set_postfix({"status": "failed", "total": stats.frames_generated})
            continue

        # Resize to match standard frame size (API may return different sizes)
        new_image = resize_to_size(new_image, frame_size, verbose=args.verbose)

        # Save frame
        frame_path = images_dir / f"frame_{frame_num:03d}"
        try:
            file_size = save_data_uri(new_image, frame_path)
            stats.frames_generated += 1
            current_image = new_image

            progress.set_postfix({
                "size": f"{file_size // 1024}KB",
                "tokens": stats.total_tokens,
            })

        except Exception as e:
            print(f"\n❌ Failed to save frame {frame_num}: {e}")
            stats.frames_failed += 1

    # Generate video
    if stats.frames_generated > 0:
        video_path = run_dir / "animation.mp4"
        generate_video(images_dir, video_path, args.fps)

    # Print report
    print(stats.report())

    # Save report to file
    report_path = run_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write(f"Image Loop Generation Report\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Input image: {args.image}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Top P: {args.top_p}\n")
        f.write(f"Size: {args.size} ({frame_size[0]}x{frame_size[1]})\n")
        f.write(f"Requested frames: {args.frames}\n")
        f.write(stats.report())

    print(f"\n📝 Report saved: {report_path}")

    return 0 if stats.frames_generated > 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Generate iterative image animations using AI models via OpenRouter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with preset mode
  uv run src/image_loop.py --image photo.jpg --mode zoom-out --frames 10

  # Custom prompt
  uv run src/image_loop.py --image photo.jpg --mode custom --prompt "Age this person by 5 years"

  # Specify model and output
  uv run src/image_loop.py --image photo.jpg --mode evolve --model flux-pro --output ./renders

  # Continue an existing run with 5 more frames
  uv run src/image_loop.py --continue output/run_flux-pro_zoom-out_1218_1234_abcd --frames 5

  # Continue with a different prompt/mode
  uv run src/image_loop.py --continue output/run_flux-pro_zoom-out_1218_1234_abcd --mode dramatic --frames 5

Available modes:
  """ + ", ".join(sorted(PROMPTS.keys())) + """

Available models:
  """ + ", ".join(f"{k} ({v})" for k, v in MODELS.items())
    )

    parser.add_argument(
        "--image", "-i",
        help="Path to the input image",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=list(PROMPTS.keys()) + ["custom"],
        help="Generation mode (preset prompt) or 'custom' for custom prompt",
    )
    parser.add_argument(
        "--prompt", "-p",
        help="Custom prompt (required when mode is 'custom')",
    )
    parser.add_argument(
        "--frames", "-n",
        type=int,
        default=10,
        help="Number of frames to generate (default: 10)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model to use. Shortcuts: {', '.join(MODELS.keys())}. Or full model ID. Default: flux-pro",
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        dest="top_p",
        help="Top-p sampling (default: 0.9)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Frames per second for output video (default: 1)",
    )
    parser.add_argument(
        "--size", "-s",
        default=DEFAULT_SIZE,
        choices=list(STANDARD_SIZES.keys()),
        help=f"Standard size for frames. Options: {', '.join(f'{k} {v}' for k, v in STANDARD_SIZES.items())}. Default: {DEFAULT_SIZE}",
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="List all available preset modes and exit",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output (show API response details)",
    )
    parser.add_argument(
        "--continue", "-c",
        dest="continue_run",
        help="Continue from an existing run directory (adds more frames)",
    )

    args = parser.parse_args()

    # Handle list commands
    if args.list_modes:
        print("Available modes:\n")
        for name, prompt in sorted(PROMPTS.items()):
            print(f"  {name:15} - {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
        return 0

    if args.list_models:
        print("Available models:\n")
        for short, full in MODELS.items():
            print(f"  {short:15} -> {full}")
        print("\nYou can also use any full OpenRouter model ID.")
        return 0

    # Handle continue mode
    if args.continue_run:
        return asyncio.run(run_continue(args))

    # Validate required args for new generation
    if not args.image:
        parser.error("--image/-i is required for new generation (or use --continue)")
    if not args.mode:
        parser.error("--mode/-m is required for generation")

    # Run the generation
    return asyncio.run(run_generation(args))


if __name__ == "__main__":
    sys.exit(main())

