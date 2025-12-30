# /// script
# dependencies = [
#   "httpx",
#   "Pillow",
#   "pyyaml",
# ]
# ///

"""
Generate images from text using multiple AI models via OpenRouter.

Run with: uv run src/generate_from_text.py "your prompt here"
"""

import argparse
import asyncio
import base64
import io
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
import yaml
from PIL import Image

# Available models for image generation
MODELS = {
    "flux-pro": "black-forest-labs/flux.2-pro",
    "gemini-flash-image": "google/gemini-2.5-flash-image",
    "riverflow": "sourceful/riverflow-v2-standard-preview",
}

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/responses"


def load_api_key() -> str:
    """Load OpenRouter API key from environment or secrets.yaml."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return api_key

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


async def fetch_image_url(url: str) -> str:
    """Fetch an image from URL and return as data URI."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "image/png")
        encoded = base64.b64encode(response.content).decode("utf-8")
        return f"data:{content_type};base64,{encoded}"


def save_data_uri(data_uri: str, output_path: Path) -> int:
    """Save a data URI as a PNG file. Returns file size in bytes."""
    _, encoded = data_uri.split(",", 1)
    image_data = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_data))
    file_path = output_path.with_suffix(".png")
    image.save(file_path, format="PNG", optimize=False)
    return len(image_data)


async def generate_image_from_text(
    prompt: str,
    model: str,
    api_key: str,
    verbose: bool = False,
) -> tuple[str | None, dict]:
    """
    Generate an image from text using the OpenRouter responses API.

    Returns:
        Tuple of (image_data_uri or None, usage_dict)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ai-feedback-loops",
        "X-Title": "Text to Image Generator",
    }

    payload = {
        "model": model,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Generate an image: {prompt}",
                    },
                ],
            }
        ],
        "temperature": 0.7,
        "top_p": 0.9,
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
                print(f"  ❌ API Error ({response.status_code}): {error_text[:200]}")
                return None, {}

            result = response.json()
            
            if verbose:
                import json
                print(f"  📡 Response keys: {list(result.keys())}")
                if result.get("usage"):
                    print(f"  📊 Usage: {json.dumps(result['usage'], indent=2)}")

            usage = result.get("usage", {})

            # Extract the generated image from the response
            output = result.get("output", [])
            for item in output:
                item_type = item.get("type")
                
                if item_type == "message":
                    content = item.get("content", [])
                    for part in content:
                        part_type = part.get("type")
                        if part_type in ("output_image", "image"):
                            image_url = part.get("image_url") or part.get("url")
                            if image_url:
                                if image_url.startswith("http"):
                                    return await fetch_image_url(image_url), usage
                                return image_url, usage

                        if part_type == "image" and part.get("data"):
                            mime = part.get("mime_type", "image/png")
                            return f"data:{mime};base64,{part['data']}", usage

                        if part_type == "image_generation_call" and part.get("result"):
                            img_result = part.get("result")
                            if isinstance(img_result, str):
                                if img_result.startswith("http"):
                                    return await fetch_image_url(img_result), usage
                                return img_result, usage

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

            # Check output_text for a URL
            output_text = result.get("output_text", "")
            if output_text:
                import re
                url_pattern = r'https?://[^\s<>"\']+\.(?:png|jpg|jpeg|gif|webp)[^\s<>"\']*'
                urls = re.findall(url_pattern, output_text, re.IGNORECASE)
                if urls:
                    return await fetch_image_url(urls[0]), usage

            # Check for error
            error = result.get("error")
            if error:
                error_msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
                print(f"  ⚠️  Model error: {error_msg[:150]}")
                return None, usage

            # Extract text response
            model_text = output_text
            if not model_text:
                for item in output:
                    if item.get("type") == "message":
                        content = item.get("content", [])
                        for part in content:
                            if part.get("type") in ("output_text", "text"):
                                model_text = part.get("text", "")
                                break
                        if model_text:
                            break

            print(f"  ⚠️  No image generated.")
            if model_text:
                text_preview = model_text[:200] + "..." if len(model_text) > 200 else model_text
                print(f"      Model said: {text_preview}")
            return None, usage

        except httpx.TimeoutException:
            print("  ❌ Request timed out")
            return None, {}
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None, {}


async def main_async(args):
    """Generate images from all models."""
    try:
        api_key = load_api_key()
    except ValueError as e:
        print(f"❌ {e}")
        return 1

    prompt = args.prompt
    print(f"📝 Prompt: {prompt}\n")

    # Determine which models to use
    if args.model:
        model_list = [args.model]
    else:
        model_list = list(MODELS.keys())

    # Set up output directory
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    output_dir = Path(args.output) / f"text2img_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output: {output_dir}\n")

    results = {}
    
    for model_short in model_list:
        model_full = MODELS.get(model_short, model_short)
        print(f"🤖 {model_short} ({model_full})")
        
        image_data, usage = await generate_image_from_text(
            prompt=prompt,
            model=model_full,
            api_key=api_key,
            verbose=args.verbose,
        )
        
        if image_data:
            output_path = output_dir / f"{model_short}"
            file_size = save_data_uri(image_data, output_path)
            print(f"  ✅ Saved: {output_path}.png ({file_size // 1024}KB)")
            results[model_short] = {
                "success": True,
                "path": f"{output_path}.png",
                "size": file_size,
                "cost": usage.get("cost", 0),
            }
        else:
            print(f"  ❌ Failed to generate image")
            results[model_short] = {"success": False}
        
        print()

    # Summary
    print("=" * 50)
    print("📊 Summary")
    print("=" * 50)
    
    successful = sum(1 for r in results.values() if r.get("success"))
    total_cost = sum(r.get("cost", 0) for r in results.values())
    
    print(f"✅ Successful: {successful}/{len(results)}")
    print(f"💰 Total cost: ${total_cost:.4f}")
    print(f"📁 Output directory: {output_dir}")

    # Save prompt to file
    prompt_file = output_dir / "prompt.txt"
    with open(prompt_file, "w") as f:
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Prompt: {prompt}\n\n")
        f.write("Results:\n")
        for model, result in results.items():
            status = "✅" if result.get("success") else "❌"
            f.write(f"  {status} {model}\n")
    
    return 0 if successful > 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from text using multiple AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Generate with all models
  uv run src/generate_from_text.py "A cat wearing a top hat"

  # Generate with a specific model
  uv run src/generate_from_text.py "A futuristic city" --model flux-pro

Available models:
  {', '.join(f'{k} ({v})' for k, v in MODELS.items())}
"""
    )

    parser.add_argument(
        "prompt",
        help="Text description of the image to generate",
    )
    parser.add_argument(
        "--model", "-m",
        help=f"Specific model to use. If not specified, uses all models. Options: {', '.join(MODELS.keys())}",
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed API responses",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:\n")
        for short, full in MODELS.items():
            print(f"  {short:20} -> {full}")
        return 0

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())

