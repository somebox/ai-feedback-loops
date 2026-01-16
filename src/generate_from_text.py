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
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from imageloop import api, settings, storage


async def main_async(args):
    """Generate images from all models."""
    try:
        api_key = storage.load_api_key()
    except ValueError as e:
        print(f"❌ {e}")
        return 1

    prompt = args.prompt
    print(f"📝 Prompt: {prompt}\n")

    # Load settings
    cfg = settings.load_settings()
    models_dict = cfg.get("models", {})

    # Determine which models to use
    if args.model:
        model_list = [args.model]
    else:
        model_list = list(models_dict.keys())

    # Set up output directory
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    output_dir = Path(args.output) / f"text2img_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output: {output_dir}\n")

    created_at = datetime.now().isoformat()
    start_time = time.time()
    results_list = []
    
    for model_short in model_list:
        model_full = settings.get_model(model_short, cfg)
        print(f"🤖 {model_short} ({model_full})")
        
        # Track timing for this model
        model_start_time = time.time()
        
        gen_result = await api.generate_from_text(
            prompt=prompt,
            model=model_full,
            api_key=api_key,
            verbose=args.verbose,
        )
        
        model_duration = time.time() - model_start_time
        usage = gen_result.get("usage", {})
        
        result_entry = {
            "model_short": model_short,
            "model_full": model_full,
            "success": gen_result.get("success", False),
            "duration_seconds": round(model_duration, 2),
            "usage": {
                "input_tokens": usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0) or usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "cost": usage.get("cost", 0.0),
            },
        }
        
        if gen_result["success"]:
            output_path = output_dir / f"{model_short}"
            file_size = storage.save_data_uri(gen_result["image"], output_path)
            print(f"  ✅ Saved: {output_path}.png ({file_size // 1024}KB)")
            result_entry["file"] = f"{model_short}.png"
            result_entry["file_size_bytes"] = file_size
            result_entry["cost"] = usage.get("cost", 0.0)
        else:
            print(f"  ❌ Failed to generate image")
            result_entry["cost"] = 0.0
            if gen_result.get("error"):
                result_entry["error"] = str(gen_result["error"])
        
        results_list.append(result_entry)
        print()

    # Calculate totals
    total_time = time.time() - start_time
    successful = sum(1 for r in results_list if r.get("success"))
    failed = len(results_list) - successful
    total_cost = sum(r.get("cost", 0) for r in results_list)
    
    # Determine status
    if failed == 0:
        status = "completed"
    elif successful == 0:
        status = "failed"
    else:
        status = "partial"

    # Create JSON structure
    run_data = {
        "summary": {
            "created": created_at,
            "type": "text-to-image",
            "prompt": prompt,
            "models_requested": model_list,
            "models_successful": successful,
            "models_failed": failed,
            "total_cost": f"${total_cost:.4f}",
            "total_time": f"{total_time:.1f}s",
            "status": status,
        },
        "config": {
            "prompt": prompt,
            "output_dir": str(output_dir),
        },
        "results": results_list,
    }

    # Save JSON artifact
    run_json_path = output_dir / "run.json"
    with open(run_json_path, "w") as f:
        json.dump(run_data, f, indent=2)

    # Summary
    print("=" * 50)
    print("📊 Summary")
    print("=" * 50)
    
    print(f"✅ Successful: {successful}/{len(results_list)}")
    print(f"💰 Total cost: ${total_cost:.4f}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"📁 Output directory: {output_dir}")

    # Save prompt to file (keep for backward compatibility)
    prompt_file = output_dir / "prompt.txt"
    with open(prompt_file, "w") as f:
        f.write(f"Generated: {created_at}\n")
        f.write(f"Prompt: {prompt}\n\n")
        f.write("Results:\n")
        for result in results_list:
            status = "✅" if result.get("success") else "❌"
            f.write(f"  {status} {result['model_short']}\n")
    
    return 0 if successful > 0 else 1


def main():
    # Load settings for model list
    cfg = settings.load_settings()
    models_dict = cfg.get("models", {})
    
    parser = argparse.ArgumentParser(
        description="Generate images from text using multiple AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Generate with all models
  uv run src/generate_from_text.py "A cat wearing a top hat"

  # Generate with a specific model
  uv run src/generate_from_text.py "A futuristic city" --model flux-pro

  # Read prompt from file
  uv run src/generate_from_text.py --file prompt.txt --model flux-pro

  # Read prompt from stdin (pipe or redirect)
  echo "A sunset over mountains" | uv run src/generate_from_text.py --model flux-pro
  cat prompt.txt | uv run src/generate_from_text.py --model flux-pro

Available models:
  {', '.join(f'{k} ({v})' for k, v in models_dict.items())}
"""
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="Text description of the image to generate (or use --file or pipe from stdin)",
    )
    parser.add_argument(
        "--model", "-m",
        help=f"Specific model to use. If not specified, uses all models. Options: {', '.join(models_dict.keys())}",
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
    parser.add_argument(
        "--file", "-f",
        type=Path,
        help="Read prompt from file instead of argument or stdin",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:\n")
        for short, full in models_dict.items():
            print(f"  {short:20} -> {full}")
        return 0

    # Determine prompt source: file > argument > stdin
    prompt = args.prompt
    if not prompt:
        if args.file:
            if not args.file.exists():
                parser.error(f"File not found: {args.file}")
            prompt = args.file.read_text().strip()
        elif not sys.stdin.isatty():
            # Read from stdin if it's piped/redirected
            prompt = sys.stdin.read().strip()
        else:
            parser.error("prompt is required (provide as argument, use --file, or pipe from stdin)")
    
    if not prompt:
        parser.error("prompt cannot be empty")
    
    # Update args with the resolved prompt
    args.prompt = prompt

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
