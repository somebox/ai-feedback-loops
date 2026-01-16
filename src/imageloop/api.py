"""OpenRouter API client for image generation."""

import asyncio
import base64
import re
import time
from pathlib import Path

import httpx

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/responses"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


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
    seed: int = None,
    verbose: bool = False,
    timeout: float = 120.0,
) -> dict:
    """
    Generate a new image using the OpenRouter responses API.

    Returns:
        Dict with keys:
            - success: bool
            - image: str | None (data URI)
            - usage: dict (token/cost info)
            - response: dict (raw API response metadata)
            - error: str | None
            - model_text: str | None (any text the model returned)
            - duration: float (seconds)
    """
    start_time = time.time()
    
    def make_result(success, image=None, usage=None, response=None, error=None, model_text=None):
        return {
            "success": success,
            "image": image,
            "usage": usage or {},
            "response": response or {},
            "error": error,
            "model_text": model_text,
            "duration": time.time() - start_time,
        }
    
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
    
    # Add seed if provided (for reproducibility)
    if seed is not None:
        payload["seed"] = seed

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
            )

            if response.status_code != 200:
                error_text = response.text
                print(f"\n❌ API Error ({response.status_code}): {error_text[:200]}")
                return make_result(False, error=f"HTTP {response.status_code}: {error_text[:200]}")

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
                                    img = await fetch_image_url(image_url)
                                    return make_result(True, image=img, usage=usage, response=result)
                                return make_result(True, image=image_url, usage=usage, response=result)

                        # Some models return base64 directly
                        if part_type == "image" and part.get("data"):
                            mime = part.get("mime_type", "image/png")
                            img = f"data:{mime};base64,{part['data']}"
                            return make_result(True, image=img, usage=usage, response=result)

                        # Check for image_generation_call type
                        if part_type == "image_generation_call" and part.get("result"):
                            img_result = part.get("result")
                            if isinstance(img_result, str):
                                if img_result.startswith("http"):
                                    img = await fetch_image_url(img_result)
                                    return make_result(True, image=img, usage=usage, response=result)
                                return make_result(True, image=img_result, usage=usage, response=result)

                # Handle image_generation_call at output level
                if item_type == "image_generation_call":
                    img_result = item.get("result")
                    if img_result:
                        if isinstance(img_result, str):
                            if img_result.startswith("http"):
                                img = await fetch_image_url(img_result)
                                return make_result(True, image=img, usage=usage, response=result)
                            return make_result(True, image=img_result, usage=usage, response=result)

            # Check for direct image in result
            if "image" in result:
                img = result["image"]
                if isinstance(img, str):
                    if img.startswith("http"):
                        fetched = await fetch_image_url(img)
                        return make_result(True, image=fetched, usage=usage, response=result)
                    elif img.startswith("data:"):
                        return make_result(True, image=img, usage=usage, response=result)

            # Check output_text for a URL (some models return image URL in text)
            output_text = result.get("output_text", "")
            if output_text:
                # Look for URLs in the text
                url_pattern = r'https?://[^\s<>"\']+\.(?:png|jpg|jpeg|gif|webp)[^\s<>"\']*'
                urls = re.findall(url_pattern, output_text, re.IGNORECASE)
                if urls:
                    img = await fetch_image_url(urls[0])
                    return make_result(True, image=img, usage=usage, response=result)

            # Check for error in response
            error = result.get("error")
            if error:
                error_msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
                print(f"\n⚠️  Model error: {error_msg[:150]}")
                return make_result(False, usage=usage, response=result, error=error_msg)

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
            return make_result(False, usage=usage, response=result, model_text=model_text, error="No image in response")

        except httpx.TimeoutException:
            print("\n❌ Request timed out")
            return make_result(False, error="Request timed out")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return make_result(False, error=str(e))


def fetch_image_models(api_key: str = None) -> list[dict]:
    """Fetch available image generation models from OpenRouter API."""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(OPENROUTER_MODELS_URL)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as e:
        print(f"❌ Failed to fetch models: {e}")
        return []
    
    models = data.get("data", [])
    
    # Filter for models that support image output (can generate images)
    image_models = []
    for model in models:
        arch = model.get("architecture", {})
        output_modalities = arch.get("output_modalities", [])
        
        # Must produce image output
        if "image" in output_modalities:
            image_models.append(model)
    
    return image_models


def verify_model_exists(model_id: str, api_key: str) -> bool:
    """Check if a model exists by querying its parameters endpoint."""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                f"https://openrouter.ai/api/v1/parameters/{model_id}",
                headers=headers
            )
            return response.status_code == 200
    except Exception:
        return False


async def generate_from_text(
    prompt: str,
    model: str,
    api_key: str,
    verbose: bool = False,
    timeout: float = 120.0,
) -> dict:
    """
    Generate an image from text using the OpenRouter responses API.

    Returns:
        Dict with keys:
            - success: bool
            - image: str | None (data URI)
            - usage: dict (token/cost info)
            - response: dict (raw API response metadata)
            - error: str | None
            - model_text: str | None (any text the model returned)
            - duration: float (seconds)
    """
    start_time = time.time()
    
    def make_result(success, image=None, usage=None, response=None, error=None, model_text=None):
        return {
            "success": success,
            "image": image,
            "usage": usage or {},
            "response": response or {},
            "error": error,
            "model_text": model_text,
            "duration": time.time() - start_time,
        }
    
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

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
            )

            if response.status_code != 200:
                error_text = response.text
                print(f"  ❌ API Error ({response.status_code}): {error_text[:200]}")
                return make_result(False, error=f"HTTP {response.status_code}: {error_text[:200]}")

            result = response.json()
            
            if verbose:
                import json
                print(f"  📡 Response keys: {list(result.keys())}")
                if result.get("usage"):
                    print(f"  📊 Usage: {json.dumps(result['usage'], indent=2)}")

            usage = result.get("usage", {})

            # Extract the generated image from the response (same parsing logic as generate_image)
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
                                    img = await fetch_image_url(image_url)
                                    return make_result(True, image=img, usage=usage, response=result)
                                return make_result(True, image=image_url, usage=usage, response=result)

                        if part_type == "image" and part.get("data"):
                            mime = part.get("mime_type", "image/png")
                            img = f"data:{mime};base64,{part['data']}"
                            return make_result(True, image=img, usage=usage, response=result)

                        if part_type == "image_generation_call" and part.get("result"):
                            img_result = part.get("result")
                            if isinstance(img_result, str):
                                if img_result.startswith("http"):
                                    img = await fetch_image_url(img_result)
                                    return make_result(True, image=img, usage=usage, response=result)
                                return make_result(True, image=img_result, usage=usage, response=result)

                if item_type == "image_generation_call":
                    img_result = item.get("result")
                    if img_result:
                        if isinstance(img_result, str):
                            if img_result.startswith("http"):
                                img = await fetch_image_url(img_result)
                                return make_result(True, image=img, usage=usage, response=result)
                            return make_result(True, image=img_result, usage=usage, response=result)

            # Check for direct image in result
            if "image" in result:
                img = result["image"]
                if isinstance(img, str):
                    if img.startswith("http"):
                        fetched = await fetch_image_url(img)
                        return make_result(True, image=fetched, usage=usage, response=result)
                    elif img.startswith("data:"):
                        return make_result(True, image=img, usage=usage, response=result)

            # Check output_text for a URL
            output_text = result.get("output_text", "")
            if output_text:
                url_pattern = r'https?://[^\s<>"\']+\.(?:png|jpg|jpeg|gif|webp)[^\s<>"\']*'
                urls = re.findall(url_pattern, output_text, re.IGNORECASE)
                if urls:
                    img = await fetch_image_url(urls[0])
                    return make_result(True, image=img, usage=usage, response=result)

            # Check for error
            error = result.get("error")
            if error:
                error_msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
                print(f"  ⚠️  Model error: {error_msg[:150]}")
                return make_result(False, usage=usage, response=result, error=error_msg)

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
            return make_result(False, usage=usage, response=result, model_text=model_text, error="No image in response")

        except httpx.TimeoutException:
            print("  ❌ Request timed out")
            return make_result(False, error="Request timed out")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return make_result(False, error=str(e))


def format_price(value: float, unit: str = "") -> str:
    """Format a price value nicely."""
    if value == 0:
        return "free"
    if value < 0.001:
        return f"${value:.6f}{unit}"
    if value < 0.01:
        return f"${value:.4f}{unit}"
    if value < 1:
        return f"${value:.3f}{unit}"
    return f"${value:.2f}{unit}"
