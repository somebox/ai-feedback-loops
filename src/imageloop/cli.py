"""CLI argument parsing and command dispatch."""

import argparse
import asyncio
import shlex
import signal
import sys
from datetime import datetime
from pathlib import Path
import uuid

from tqdm import tqdm

from imageloop import api, job, runlog, settings, sizing, storage
from imageloop.sizing import STANDARD_SIZES, SPECIAL_SIZES, DEFAULT_SIZE

# Global flag for graceful shutdown
_shutdown_requested = False


def handle_shutdown(signum, frame):
    """Handle interrupt signals gracefully."""
    global _shutdown_requested
    if _shutdown_requested:
        # Second interrupt - force exit
        print("\n\n⚠️  Force quit requested. Exiting immediately...")
        sys.exit(1)
    _shutdown_requested = True
    print("\n\n⚠️  Interrupt received. Finishing current frame and saving progress...")


# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


def cmd_list_modes(cfg: dict):
    """List all available preset modes."""
    prompts = cfg.get("prompts", {})
    print("Available modes:\n")
    for name, prompt in sorted(prompts.items()):
        print(f"  {name:15} - {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    return 0


def build_command_line(args, defaults: dict, script_name: str = "python src/image_loop.py") -> str:
    """Reconstruct the command line from parsed arguments, omitting defaults."""
    parts = [script_name]
    
    # Required/important arguments
    if hasattr(args, 'image') and args.image:
        parts.append(f"--image {shlex.quote(str(args.image))}")
    
    if hasattr(args, 'mode') and args.mode:
        parts.append(f"--mode {shlex.quote(args.mode)}")
    
    if hasattr(args, 'prompt') and args.prompt:
        parts.append(f"--prompt {shlex.quote(args.prompt)}")
    
    # Optional arguments (only include if different from defaults)
    if hasattr(args, 'frames') and args.frames != defaults.get("frames", 10):
        parts.append(f"--frames {args.frames}")
    
    if hasattr(args, 'model') and args.model:
        parts.append(f"--model {shlex.quote(args.model)}")
    
    if hasattr(args, 'output') and args.output != defaults.get("output_dir", "output"):
        parts.append(f"--output {shlex.quote(str(args.output))}")
    
    if hasattr(args, 'size') and args.size != defaults.get("size", "auto"):
        parts.append(f"--size {shlex.quote(args.size)}")
        if args.size == "custom" and hasattr(args, 'width') and hasattr(args, 'height'):
            if args.width and args.height:
                parts.append(f"--width {args.width} --height {args.height}")
    
    if hasattr(args, 'temperature') and args.temperature != defaults.get("temperature", 0.7):
        parts.append(f"--temperature {args.temperature}")
    
    if hasattr(args, 'top_p') and args.top_p != defaults.get("top_p", 0.9):
        parts.append(f"--top-p {args.top_p}")
    
    if hasattr(args, 'seed') and args.seed is not None:
        parts.append(f"--seed {args.seed}")
    
    if hasattr(args, 'fps') and args.fps != defaults.get("fps", 1):
        parts.append(f"--fps {args.fps}")
    
    if hasattr(args, 'format') and args.format != defaults.get("output_format", "mp4"):
        parts.append(f"--format {args.format}")
    
    if hasattr(args, 'verbose') and args.verbose:
        parts.append("--verbose")
    
    return " ".join(parts)


def cmd_list_models(cfg: dict):
    """List available image generation models with pricing."""
    from imageloop.api import fetch_image_models, verify_model_exists, format_price
    
    # Try to load API key for model verification
    try:
        api_key = storage.load_api_key()
    except ValueError:
        api_key = None
    
    print("🔍 Fetching image generation models from OpenRouter...\n")
    
    models = fetch_image_models()
    
    if models:
        # Sort by image cost (cheapest first), then by model ID
        def sort_key(m):
            pricing = m.get("pricing", {})
            image_cost = float(pricing.get("image", 0) or 0)
            return (image_cost, m.get("id", ""))
        
        models.sort(key=sort_key)
        
        print(f"Models with image output (from /api/v1/models):\n")
        print(f"{'Model ID':<50} {'Per Image':<12} {'Per 1K Tokens':<15}")
        print("-" * 78)
        
        for model in models:
            model_id = model.get("id", "unknown")
            pricing = model.get("pricing", {})
            
            # Image cost is per generated image
            image_cost = float(pricing.get("image", 0) or 0)
            
            # Token costs (prompt + completion average, per 1K tokens)
            prompt_cost = float(pricing.get("prompt", 0) or 0) * 1000
            completion_cost = float(pricing.get("completion", 0) or 0) * 1000
            token_cost = (prompt_cost + completion_cost) / 2 if (prompt_cost or completion_cost) else 0
            
            image_str = format_price(image_cost, "/img") if image_cost else "-"
            token_str = format_price(token_cost, "/1K") if token_cost else "-"
            
            print(f"{model_id:<50} {image_str:<12} {token_str:<15}")
        
        print("-" * 78)
    else:
        print("No models with image output found in API.\n")
    
    # Verify and show configured shortcuts
    print("\nConfigured shortcuts:")
    models_dict = cfg.get("models", {})
    for short, full in models_dict.items():
        if api_key:
            exists = verify_model_exists(full, api_key)
            status = "✓" if exists else "✗"
        else:
            status = "?"
        print(f"  {status} {short:18} -> {full}")
    
    if not api_key:
        print("\n  (Set API key to verify model availability)")
    
    print("\nNote: Some image models use OpenRouter's image generation endpoint")
    print("and don't appear in the standard models API. Browse all at:")
    print("  https://openrouter.ai/models?output_modalities=image")
    
    return 0


async def cmd_continue(args, cfg: dict):
    """Continue an existing generation run."""
    global _shutdown_requested
    
    # Load API key
    try:
        api_key = storage.load_api_key()
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
    result = job.find_last_frame(images_dir)
    if result is None:
        print(f"❌ No frames found in {images_dir}")
        return 1
    
    last_frame_num, last_frame_path = result
    print(f"📁 Continuing from: {run_dir}")
    print(f"🖼️  Last frame: {last_frame_path.name} (frame {last_frame_num})")

    # Load existing run log or fall back to report.txt
    run_log_path = run_dir / "run.json"
    if run_log_path.exists():
        run_log = runlog.RunLog.load(run_log_path)
        saved_settings = run_log.config
    else:
        # Fall back to parsing report.txt for older runs
        report_path = run_dir / "report.txt"
        saved_settings = runlog.parse_legacy_report(report_path)
        run_log = runlog.RunLog(run_dir=run_dir)
        # Populate from parsed settings
        run_log.set_config(**saved_settings)
    
    # Build continue command line
    run_path = run_dir
    # Use relative path if possible for cleaner output
    try:
        run_path = run_path.relative_to(Path.cwd())
    except ValueError:
        pass
    
    # Build continue command from current args, replacing --image with --continue
    continue_parts = ["python", "src/image_loop.py", f"--continue {shlex.quote(str(run_path))}"]
    
    # Add other args that differ from defaults or saved settings
    defaults = cfg.get("defaults", {})
    if args.frames != saved_settings.get("requested_frames", defaults.get("frames", 10)):
        continue_parts.append(f"--frames {args.frames}")
    
    if args.mode and args.mode != saved_settings.get("mode"):
        continue_parts.append(f"--mode {shlex.quote(args.mode)}")
        if args.prompt:
            continue_parts.append(f"--prompt {shlex.quote(args.prompt)}")
    
    if args.model:
        continue_parts.append(f"--model {shlex.quote(args.model)}")
    
    if args.temperature != saved_settings.get("temperature", defaults.get("temperature", 0.7)):
        continue_parts.append(f"--temperature {args.temperature}")
    
    if args.top_p != saved_settings.get("top_p", defaults.get("top_p", 0.9)):
        continue_parts.append(f"--top-p {args.top_p}")
    
    if args.seed is not None:
        continue_parts.append(f"--seed {args.seed}")
    
    if args.fps != saved_settings.get("fps", defaults.get("fps", 1)):
        continue_parts.append(f"--fps {args.fps}")
    
    if args.format != saved_settings.get("output_format", defaults.get("output_format", "mp4")):
        continue_parts.append(f"--format {args.format}")
    
    if args.verbose:
        continue_parts.append("--verbose")
    
    continue_command = " ".join(continue_parts)
    
    # Model priority: explicit CLI arg > saved from log > default
    if args.model:
        model = settings.get_model(args.model, cfg)
    elif saved_settings.get("model"):
        model = saved_settings["model"]
    else:
        default_model = cfg.get("defaults", {}).get("model", "flux-pro")
        model = settings.get_model(default_model, cfg)
    
    # Get prompt - prefer CLI args, fall back to saved
    if args.mode:
        prompt = settings.get_prompt(args.mode, args.prompt, cfg)
        mode = args.mode
    elif saved_settings.get("prompt"):
        prompt = saved_settings["prompt"]
        mode = saved_settings.get("mode", "custom")
    else:
        print("❌ No mode specified and couldn't find prompt in run log")
        return 1

    defaults = cfg.get("defaults", {})
    temperature = args.temperature if args.temperature != defaults.get("temperature", 0.7) else saved_settings.get("temperature", defaults.get("temperature", 0.7))
    top_p = args.top_p if args.top_p != defaults.get("top_p", 0.9) else saved_settings.get("top_p", defaults.get("top_p", 0.9))

    # Get frame size from saved settings or detect from last frame
    frame_dims = saved_settings.get("frame_dimensions")
    if frame_dims:
        frame_size = (frame_dims["width"], frame_dims["height"])
    elif saved_settings.get("size") and saved_settings["size"] in STANDARD_SIZES:
        frame_size = STANDARD_SIZES[saved_settings["size"]]
    else:
        # Detect size from last frame
        from PIL import Image
        last_image = Image.open(last_frame_path)
        frame_size = last_image.size
        last_image.close()
    
    print(f"🤖 Model: {model}")
    print(f"💬 Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"📐 Frame size: {frame_size[0]}x{frame_size[1]}")

    # Load the last frame as current image
    print(f"📷 Loading last frame...")
    current_image = storage.image_to_data_uri(last_frame_path)

    # Start a new session in the run log
    run_log.start_session(is_continuation=True)
    run_log._current_session["frames_requested"] = args.frames
    
    # Save state so we can continue if interrupted
    run_log._auto_save()

    # Generation loop
    print(f"\n🎬 Generating {args.frames} more frames...\n")

    progress = tqdm(range(args.frames), desc="Generating", unit="frame")
    interrupted = False

    for i in progress:
        # Check for shutdown request
        if _shutdown_requested:
            interrupted = True
            progress.close()
            print(f"\n⏹️  Stopping after frame {last_frame_num + i} (interrupt requested)")
            break
            
        frame_num = last_frame_num + i + 1

        gen_result = await api.generate_image(
            prompt=prompt,
            image_data_uri=current_image,
            model=model,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            seed=args.seed,
            verbose=args.verbose,
        )

        if not gen_result["success"]:
            run_log.log_frame(
                frame_number=frame_num,
                success=False,
                usage=gen_result.get("usage"),
                api_response=gen_result.get("response"),
                duration_seconds=gen_result.get("duration"),
                error=gen_result.get("error"),
                model_text=gen_result.get("model_text"),
            )
            progress.set_postfix({"status": "failed", "total": run_log.stats["frames_generated"]})
            continue

        new_image = gen_result["image"]
        
        # Resize to match existing frame size (use crop to preserve aspect ratio)
        new_image = sizing.resize_to_size(new_image, frame_size, verbose=args.verbose, mode="crop")

        # Save frame
        frame_stem = images_dir / f"frame_{frame_num:03d}"
        frame_path = frame_stem.with_suffix(".png")
        try:
            file_size = storage.save_data_uri(new_image, frame_stem)
            
            # Log successful frame (auto-saves after each frame)
            run_log.log_frame(
                frame_number=frame_num,
                success=True,
                usage=gen_result.get("usage"),
                api_response=gen_result.get("response"),
                file_path=str(frame_path),
                file_size_bytes=file_size,
                output_dimensions=frame_size,
                duration_seconds=gen_result.get("duration"),
            )
            
            current_image = new_image

            progress.set_postfix({
                "size": f"{file_size // 1024}KB",
                "cost": f"${run_log.stats['total_cost']:.3f}",
            })

        except Exception as e:
            print(f"\n❌ Failed to save frame {frame_num}: {e}")
            run_log.log_frame(
                frame_number=frame_num,
                success=False,
                usage=gen_result.get("usage"),
                duration_seconds=gen_result.get("duration"),
                error=f"Failed to save: {e}",
            )

    # End session (mark as interrupted if needed)
    if interrupted:
        run_log.mark_interrupted()
    else:
        run_log.end_session()

    # Regenerate outputs with all frames
    # Use saved format/fps from config to match original run, unless explicitly overridden
    # Check if user explicitly overrode format by comparing to default
    defaults = cfg.get("defaults", {})
    saved_format = saved_settings.get("output_format", defaults.get("output_format", "mp4"))
    default_format = defaults.get("output_format", "mp4")
    
    # If args.format matches default but saved format is different, user didn't override - use saved
    # If args.format is different from default, user explicitly set it - use args.format
    if args.format == default_format and args.format != saved_format:
        output_format = saved_format
    else:
        output_format = args.format
    
    # Same logic for fps
    saved_fps = saved_settings.get("fps", defaults.get("fps", 1))
    default_fps = defaults.get("fps", 1)
    if args.fps == default_fps and args.fps != saved_fps:
        output_fps = saved_fps
    else:
        output_fps = args.fps
    
    if run_log.stats["frames_generated"] > 0:
        print(f"\n🎬 Regenerating outputs ({output_format}) with all {run_log.stats['total_frames']} frames...")
        job.generate_outputs(images_dir, run_dir, output_fps, output_format)

    # Print and save report
    run_log.print_summary(show_continue_command=True, continue_command=continue_command)
    log_path = run_log.save()
    print(f"\n📝 Run log saved: {log_path}")

    # Show total frame count
    total_result = job.find_last_frame(images_dir)
    if total_result:
        print(f"📊 Total frames now: {total_result[0] + 1}")

    return 0 if run_log.stats["frames_generated"] > 0 else 1


async def cmd_generate(args, cfg: dict):
    """Main generation loop."""
    global _shutdown_requested
    
    # Load API key
    try:
        api_key = storage.load_api_key()
    except ValueError as e:
        print(f"❌ {e}")
        return 1

    # Resolve model name
    defaults = cfg.get("defaults", {})
    model_arg = args.model or defaults.get("model", "flux-pro")
    model = settings.get_model(model_arg, cfg)
    print(f"🤖 Model: {model}")

    # Load input image
    input_path = Path(args.image)
    if not input_path.exists():
        print(f"❌ Image not found: {input_path}")
        return 1

    print(f"📷 Loading image: {input_path}")
    current_image = storage.image_to_data_uri(input_path)

    # Handle custom size
    custom_size = None
    if args.size == "custom":
        if not args.width or not args.height:
            print("❌ --size custom requires both --width and --height")
            return 1
        custom_size = (args.width, args.height)

    # Standardize to target size for consistent frames
    current_image, frame_size, size_name = sizing.standardize_image(current_image, args.size, custom_size)
    print(f"📐 Standardized to {frame_size[0]}x{frame_size[1]} ({size_name})")

    # Get prompt
    try:
        prompt = settings.get_prompt(args.mode, args.prompt, cfg)
    except ValueError as e:
        print(f"❌ {e}")
        return 1

    print(f"💬 Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    # Set up output directory
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_id = uuid.uuid4().hex[:4]
    model_short = settings.get_model_short_name(model, cfg)
    mode_short = args.mode[:15]

    output_base = Path(args.output)
    run_dir = output_base / f"run_{model_short}_{mode_short}_{timestamp}_{run_id}"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output: {run_dir}")

    # Build command line for reference
    command_line = build_command_line(args, defaults, script_name="python src/image_loop.py")
    
    # Initialize run log
    run_log = runlog.RunLog(run_dir=run_dir)
    run_log.set_config(
        input_image=str(input_path.absolute()),
        model=model,
        mode=args.mode,
        prompt=prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        size=size_name,
        frame_dimensions={"width": frame_size[0], "height": frame_size[1]},
        requested_frames=args.frames,
        fps=args.fps,
        output_format=args.format,
        command_line=command_line,
    )

    # Save initial frame
    initial_frame_path = images_dir / "frame_000.png"
    storage.save_data_uri(current_image, images_dir / "frame_000")
    print("✅ Saved initial frame (frame_000.png)")
    
    # Log initial frame (frame 0 is the input, not generated)
    run_log.frames.append({
        "frame_number": 0,
        "timestamp": datetime.now().isoformat(),
        "success": True,
        "is_input": True,
        "file": str(initial_frame_path),
        "dimensions": {"width": frame_size[0], "height": frame_size[1]},
    })

    # Start generation session
    run_log.start_session(is_continuation=False)
    run_log._current_session["frames_requested"] = args.frames
    
    # Save initial state so we can continue if interrupted
    run_log._auto_save()

    # Generation loop
    print(f"\n🎬 Generating {args.frames} frames...\n")

    progress = tqdm(range(args.frames), desc="Generating", unit="frame")
    interrupted = False

    for i in progress:
        # Check for shutdown request
        if _shutdown_requested:
            interrupted = True
            progress.close()
            print(f"\n⏹️  Stopping after frame {i} (interrupt requested)")
            break
            
        frame_num = i + 1

        gen_result = await api.generate_image(
            prompt=prompt,
            image_data_uri=current_image,
            model=model,
            api_key=api_key,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            verbose=args.verbose,
        )

        if not gen_result["success"]:
            run_log.log_frame(
                frame_number=frame_num,
                success=False,
                usage=gen_result.get("usage"),
                api_response=gen_result.get("response"),
                duration_seconds=gen_result.get("duration"),
                error=gen_result.get("error"),
                model_text=gen_result.get("model_text"),
            )
            progress.set_postfix({"status": "failed", "total": run_log.stats["frames_generated"]})
            continue

        new_image = gen_result["image"]

        # Resize to match standard frame size (use crop to preserve aspect ratio)
        new_image = sizing.resize_to_size(new_image, frame_size, verbose=args.verbose, mode="crop")

        # Save frame
        frame_stem = images_dir / f"frame_{frame_num:03d}"
        frame_path = frame_stem.with_suffix(".png")
        try:
            file_size = storage.save_data_uri(new_image, frame_stem)
            
            # Log successful frame (auto-saves after each frame)
            run_log.log_frame(
                frame_number=frame_num,
                success=True,
                usage=gen_result.get("usage"),
                api_response=gen_result.get("response"),
                file_path=str(frame_path),
                file_size_bytes=file_size,
                output_dimensions=frame_size,
                duration_seconds=gen_result.get("duration"),
            )
            
            current_image = new_image

            progress.set_postfix({
                "size": f"{file_size // 1024}KB",
                "cost": f"${run_log.stats['total_cost']:.3f}",
            })

        except Exception as e:
            print(f"\n❌ Failed to save frame {frame_num}: {e}")
            run_log.log_frame(
                frame_number=frame_num,
                success=False,
                usage=gen_result.get("usage"),
                duration_seconds=gen_result.get("duration"),
                error=f"Failed to save: {e}",
            )

    # End session (mark as interrupted if needed)
    if interrupted:
        run_log.mark_interrupted()
    else:
        run_log.end_session()

    # Generate outputs (MP4, GIF, or both)
    if run_log.stats["frames_generated"] > 0:
        job.generate_outputs(images_dir, run_dir, args.fps, args.format)

    # Print and save report
    run_log.print_summary(show_continue_command=True)
    log_path = run_log.save()
    print(f"\n📝 Run log saved: {log_path}")

    return 0 if run_log.stats["frames_generated"] > 0 else 1


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    """Build the argument parser with settings from config."""
    prompts = cfg.get("prompts", {})
    models = cfg.get("models", {})
    defaults = cfg.get("defaults", {})
    sizes = cfg.get("sizes", {})
    
    parser = argparse.ArgumentParser(
        description="Generate iterative image animations using AI models via OpenRouter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
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
  {', '.join(sorted(prompts.keys()))}

Available models:
  {', '.join(f'{k} ({v})' for k, v in models.items())}
"""
    )

    parser.add_argument(
        "--image", "-i",
        help="Path to the input image",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=list(prompts.keys()) + ["custom"],
        help="Generation mode (preset prompt) or 'custom' for custom prompt",
    )
    parser.add_argument(
        "--prompt", "-p",
        help="Custom prompt (required when mode is 'custom')",
    )
    parser.add_argument(
        "--frames", "-n",
        type=int,
        default=defaults.get("frames", 10),
        help=f"Number of frames to generate (default: {defaults.get('frames', 10)})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model to use. Shortcuts: {', '.join(models.keys())}. Or full model ID. Default: {defaults.get('model', 'flux-pro')}",
    )
    parser.add_argument(
        "--output", "-o",
        default=defaults.get("output_dir", "output"),
        help=f"Output directory (default: {defaults.get('output_dir', 'output')})",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=defaults.get("temperature", 0.7),
        help=f"Temperature for generation (default: {defaults.get('temperature', 0.7)})",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=defaults.get("top_p", 0.9),
        dest="top_p",
        help=f"Top-p sampling (default: {defaults.get('top_p', 0.9)})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (supported by some models)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=defaults.get("fps", 1),
        help=f"Frames per second for output video/GIF (default: {defaults.get('fps', 1)})",
    )
    parser.add_argument(
        "--format", "-f",
        default=defaults.get("output_format", "mp4"),
        choices=["mp4", "gif", "both"],
        help=f"Output format: mp4, gif, or both (default: {defaults.get('output_format', 'mp4')})",
    )
    parser.add_argument(
        "--size", "-s",
        default=defaults.get("size", DEFAULT_SIZE),
        choices=SPECIAL_SIZES + list(STANDARD_SIZES.keys()),
        help=f"Frame size: auto (closest preset), preserve (keep aspect ratio), custom (--width/--height), or preset. Presets: {', '.join(f'{k} {v}' for k, v in STANDARD_SIZES.items())}. Default: {defaults.get('size', DEFAULT_SIZE)}",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Custom frame width (requires --size custom)",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Custom frame height (requires --size custom)",
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

    return parser


def main():
    """Main entry point for CLI."""
    # Load settings
    cfg = settings.load_settings()
    
    # Build parser
    parser = build_parser(cfg)
    args = parser.parse_args()

    # Handle list commands
    if args.list_modes:
        return cmd_list_modes(cfg)

    if args.list_models:
        return cmd_list_models(cfg)

    # Handle continue mode
    if args.continue_run:
        return asyncio.run(cmd_continue(args, cfg))

    # Validate required args for new generation
    if not args.image:
        parser.error("--image/-i is required for new generation (or use --continue)")
    if not args.mode:
        parser.error("--mode/-m is required for generation")

    # Run the generation
    return asyncio.run(cmd_generate(args, cfg))


if __name__ == "__main__":
    sys.exit(main())
