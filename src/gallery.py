#!/usr/bin/env python3
"""
Web gallery for viewing image loop runs.
Displays all runs as cards with thumbnails, and provides a modal viewer for animations.
"""

import argparse
import json
import mimetypes
import os
import re
import sys
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "output"


def parse_legacy_report(report_path: Path, folder: Path) -> dict | None:
    """Parse a legacy report.txt file into run data."""
    from imageloop import runlog, settings
    
    try:
        # Use runlog's parse_legacy_report to get settings
        saved_settings = runlog.parse_legacy_report(report_path)
        cfg = settings.load_settings()
        
        # Read the file for additional info
        content = report_path.read_text()
        
        # Extract fields with simple parsing
        def extract(pattern: str, default: str = "") -> str:
            match = re.search(pattern, content)
            return match.group(1).strip() if match else default
        
        # Parse date from "Generated: 2025-12-18T18:16:59.890833"
        date_str = extract(r"Generated:\s*(\S+)")
        try:
            date_obj = datetime.fromisoformat(date_str)
            date_display = date_obj.strftime("%b %d, %H:%M")
        except (ValueError, TypeError):
            date_display = date_str[:16] if date_str else "Unknown"
        
        model = saved_settings.get("model") or extract(r"Model:\s*(.+)")
        model_short = settings.get_model_short_name(model, cfg)
        mode = saved_settings.get("mode") or extract(r"Mode:\s*(.+)")
        prompt = saved_settings.get("prompt") or extract(r"Prompt:\s*(.+)")
        size = saved_settings.get("size") or extract(r"Size:\s*(.+)")
        
        # Parse stats
        total_time = extract(r"Total time:\s*([^\s]+)")
        frames_gen = extract(r"Frames generated:\s*(\d+)", "0")
        total_cost = extract(r"Total cost:\s*(\$[\d.]+)", "$0.00")
        
        # Find image frames
        images_dir = folder / "images"
        frames = []
        if images_dir.exists():
            for img in sorted(images_dir.glob("frame_*.png")):
                frame_num = int(img.stem.split("_")[1])
                frames.append({
                    "number": frame_num,
                    "file": f"output/{folder.name}/images/{img.name}",
                    "success": True,
                    "duration": None,
                    "cost": None,
                })
        
        first_frame = frames[0] if frames else None
        last_frame = frames[-1] if frames else None
        
        return {
            "folder": folder.name,
            "model": model,
            "model_short": model_short,
            "mode": mode or folder.name,
            "date": date_display,
            "date_iso": date_str,
            "status": "complete" if frames else "unknown",
            "total_frames": int(frames_gen) if frames_gen else len(frames),
            "total_cost": total_cost,
            "total_time": total_time,
            "prompt": prompt,
            "size": size,
            "dimensions": {},
            "first_frame": first_frame["file"] if first_frame else None,
            "last_frame": last_frame["file"] if last_frame else None,
            "frames": frames,
            "stats": {},
            "config": {},
            "legacy": True,
        }
    except Exception as e:
        print(f"Warning: Could not parse legacy {report_path}: {e}")
        return None


def get_runs(output_dir: Path) -> list[dict]:
    """Scan output directory for run folders with run.json or report.txt files."""
    runs = []
    
    for folder in sorted(output_dir.iterdir(), reverse=True):
        if not folder.is_dir():
            continue
        
        run_json = folder / "run.json"
        report_txt = folder / "report.txt"
        
        # Try run.json first (new format)
        if run_json.exists():
            try:
                with open(run_json) as f:
                    data = json.load(f)
                
                summary = data.get("summary", {})
                run_type = summary.get("type")
                
                # Handle text-to-image runs
                if run_type == "text-to-image" or folder.name.startswith("text2img_"):
                    from imageloop import settings
                    cfg = settings.load_settings()
                    
                    # Extract text-to-image specific data
                    prompt = summary.get("prompt", "")
                    prompt_snippet = prompt[:150] + "..." if len(prompt) > 150 else prompt
                    results = data.get("results", [])
                    config = data.get("config", {})
                    
                    # Find successful images
                    successful_results = [r for r in results if r.get("success") and r.get("file")]
                    first_image = successful_results[0] if successful_results else None
                    
                    # Get models used
                    models_used = [r.get("model_short", "unknown") for r in results]
                    models_successful = summary.get("models_successful", 0)
                    
                    # Parse date
                    created = summary.get("created", "")
                    try:
                        date_obj = datetime.fromisoformat(created)
                        date_display = date_obj.strftime("%b %d, %H:%M")
                    except (ValueError, TypeError):
                        date_display = created[:16] if created else "Unknown"
                    
                    # Build images list for modal
                    images = []
                    for result in successful_results:
                        images.append({
                            "model_short": result.get("model_short", "unknown"),
                            "model_full": result.get("model_full", ""),
                            "file": f"output/{folder.name}/{result.get('file', '')}",
                            "cost": result.get("cost", 0),
                            "duration": result.get("duration_seconds"),
                            "file_size_bytes": result.get("file_size_bytes"),
                            "usage": result.get("usage", {}),
                        })
                    
                    runs.append({
                        "folder": folder.name,
                        "type": "text-to-image",
                        "model": models_used[0] if models_used else "unknown",
                        "model_short": models_used[0] if models_used else "unknown",
                        "models": models_used,
                        "mode": "from text",
                        "date": date_display,
                        "date_iso": created,
                        "status": summary.get("status", "unknown"),
                        "total_frames": models_successful,  # Reuse for model count
                        "total_cost": summary.get("total_cost", "$0.00"),
                        "total_time": summary.get("total_time", "0.0s"),
                        "prompt": prompt,
                        "prompt_snippet": prompt_snippet,
                        "first_image": f"output/{folder.name}/{first_image.get('file')}" if first_image else None,
                        "images": images,
                        "results": results,
                        "config": config,
                        "legacy": False,
                    })
                
                # Handle image loop runs (existing logic) and prompt-loop runs
                else:
                    from imageloop import runlog, settings
                    
                    # Use RunLog.load() to parse the JSON (tested, maintained code)
                    run_log = runlog.RunLog.load(run_json)
                    cfg = settings.load_settings()
                    
                    # Extract key info from RunLog object
                    summary = run_log.to_dict().get("summary", {})
                    config = run_log.config
                    stats = run_log.stats
                    frames = run_log.frames
                    
                    # Detect prompt-loop mode
                    mode = summary.get("mode", config.get("mode", "unknown"))
                    is_prompt_loop = mode == "prompt-loop"
                    
                    # Find first and last successful frames
                    successful_frames = [f for f in frames if f.get("success") and f.get("file")]
                    first_frame = successful_frames[0] if successful_frames else None
                    last_frame = successful_frames[-1] if successful_frames else None
                    
                    # Get short model name using settings module
                    model = summary.get("model", config.get("model", "unknown"))
                    model_short = settings.get_model_short_name(model, cfg)
                    
                    # Parse date
                    created = summary.get("created", "")
                    try:
                        date_obj = datetime.fromisoformat(created)
                        date_display = date_obj.strftime("%b %d, %H:%M")
                    except (ValueError, TypeError):
                        date_display = created[:16] if created else "Unknown"
                    
                    # Determine status and progress
                    status = summary.get("status", "unknown")
                    requested = config.get("requested_frames", 0)
                    generated = stats.get("frames_generated", len(successful_frames))
                    
                    # Override status if frames don't match requested (still in progress or interrupted)
                    if requested > 0 and generated < requested:
                        if status in ("completed", "complete"):
                            status = "in_progress"  # Likely still running
                        progress_str = f"{generated}/{requested}"
                    else:
                        progress_str = None
                    
                    # Build frame data with descriptions for prompt-loop
                    frame_data = []
                    for f in frames:
                        if not f.get("file"):
                            continue
                        frame_entry = {
                            "number": f.get("frame_number"),
                            "file": f.get("file"),
                            "success": f.get("success", False),
                            "duration": f.get("duration_seconds"),
                            "cost": f.get("usage", {}).get("cost") if f.get("usage") else None,
                        }
                        # Add prompt-loop specific fields
                        if is_prompt_loop:
                            frame_entry["description"] = f.get("description")
                            frame_entry["description_file"] = f.get("description_file")
                            frame_entry["describe_cost"] = f.get("describe_usage", {}).get("cost") if f.get("describe_usage") else None
                            frame_entry["describe_duration"] = f.get("describe_duration_seconds")
                        frame_data.append(frame_entry)
                    
                    # For prompt-loop, use describe_prompt as the main prompt display
                    if is_prompt_loop:
                        describe_prompt = config.get("describe_prompt", "")
                        prompt_display = f"Describe: {describe_prompt[:100]}..." if len(describe_prompt) > 100 else f"Describe: {describe_prompt}"
                    else:
                        prompt_display = config.get("prompt", "")
                    
                    run_entry = {
                        "folder": folder.name,
                        "type": "prompt-loop" if is_prompt_loop else "image-loop",
                        "model": model,
                        "model_short": model_short,
                        "mode": mode,
                        "date": date_display,
                        "date_iso": created,
                        "status": status,
                        "progress": progress_str,
                        "total_frames": generated,
                        "requested_frames": requested,
                        "total_cost": summary.get("total_cost", f"${stats.get('total_cost', 0):.4f}"),
                        "total_time": summary.get("total_time", f"{stats.get('total_time_seconds', 0):.1f}s"),
                        "prompt": prompt_display,
                        "size": config.get("size", ""),
                        "dimensions": config.get("frame_dimensions", {}),
                        "first_frame": first_frame.get("file") if first_frame else None,
                        "last_frame": last_frame.get("file") if last_frame else None,
                        "frames": frame_data,
                        "stats": stats,
                        "config": config,
                        "legacy": False,
                    }
                    
                    # Add prompt-loop specific config
                    if is_prompt_loop:
                        run_entry["describe_mode"] = config.get("describe_mode", "detailed")
                        run_entry["describe_prompt"] = config.get("describe_prompt", "")
                    
                    runs.append(run_entry)
            except Exception as e:
                print(f"Warning: Could not parse {run_json}: {e}")
                continue
        
        # Fall back to report.txt (legacy format)
        elif report_txt.exists():
            run_data = parse_legacy_report(report_txt, folder)
            if run_data:
                runs.append(run_data)
        
        # No metadata file - try to show anyway if images exist
        else:
            images_dir = folder / "images"
            if images_dir.exists():
                frames = []
                for img in sorted(images_dir.glob("frame_*.png")):
                    frame_num = int(img.stem.split("_")[1])
                    frames.append({
                        "number": frame_num,
                        "file": f"output/{folder.name}/images/{img.name}",
                        "success": True,
                        "duration": None,
                        "cost": None,
                    })
                
                if frames:
                    runs.append({
                        "folder": folder.name,
                        "model": "unknown",
                        "model_short": "unknown",
                        "mode": folder.name,
                        "date": "Unknown",
                        "date_iso": "",
                        "status": "unknown",
                        "total_frames": len(frames),
                        "total_cost": "—",
                        "total_time": "—",
                        "prompt": "",
                        "size": "",
                        "dimensions": {},
                        "first_frame": frames[0]["file"] if frames else None,
                        "last_frame": frames[-1]["file"] if frames else None,
                        "frames": frames,
                        "stats": {},
                        "config": {},
                        "legacy": True,
                    })
    
    # Sort by date (newest first), with unknown dates at the end
    runs.sort(key=lambda r: r.get("date_iso") or "", reverse=True)
    
    return runs


def generate_html(runs: list[dict], output_dir: Path) -> str:
    """Generate the gallery HTML page."""
    
    # Generate run cards
    cards_html = ""
    for run in runs:
        run_type = run.get("type", "image-loop")
        
        status = run['status']
        if status in ("complete", "completed"):
            status_class = "status-complete"
        elif status == "in_progress":
            status_class = "status-progress"
        else:
            status_class = "status-partial"
        
        # Text-to-image cards
        if run_type == "text-to-image":
            first_thumb = f"/{run['first_image']}" if run.get('first_image') else ""
            prompt_snippet = run.get('prompt_snippet', run.get('prompt', ''))[:100]
            model_count = run.get('total_frames', len(run.get('images', [])))
            
            cards_html += f'''
        <div class="card" data-folder="{run['folder']}" onclick="openModal('{run['folder']}')">
            <div class="card-thumbnails">
                <img src="{first_thumb}" alt="Generated image" class="thumb thumb-single" loading="lazy">
            </div>
            <div class="card-info">
                <div class="card-header">
                    <span class="model-name">{run['model_short']}</span>
                    <span class="mode-badge">{run['mode']}</span>
                </div>
                <div class="card-prompt">{prompt_snippet}</div>
                <div class="card-meta">
                    <span class="date">{run['date']}</span>
                    <span class="frames">{model_count} model{'' if model_count == 1 else 's'}</span>
                    <span class="cost">{run['total_cost']}</span>
                </div>
                <div class="card-status {status_class}">{run['status']}</div>
            </div>
        </div>
        '''
        
        # Prompt loop cards
        elif run_type == "prompt-loop":
            first_thumb = f"/{run['first_frame']}" if run.get('first_frame') else ""
            last_thumb = f"/{run['last_frame']}" if run.get('last_frame') else ""
            
            cards_html += f'''
        <div class="card" data-folder="{run['folder']}" onclick="openModal('{run['folder']}')">
            <div class="card-thumbnails">
                <img src="{first_thumb}" alt="First frame" class="thumb thumb-first" loading="lazy">
                <div class="thumb-arrow">🔄</div>
                <img src="{last_thumb}" alt="Last frame" class="thumb thumb-last" loading="lazy">
            </div>
            <div class="card-info">
                <div class="card-header">
                    <span class="model-name">{run['model_short']}</span>
                    <span class="mode-badge mode-promptloop">prompt-loop</span>
                </div>
                <div class="card-meta">
                    <span class="date">{run['date']}</span>
                    <span class="frames">{run['total_frames']} frames</span>
                    <span class="cost">{run['total_cost']}</span>
                </div>
                <div class="card-status {status_class}">{run['status']}{f" ({run['progress']})" if run.get('progress') else ''}</div>
            </div>
        </div>
        '''
        
        # Image loop cards (existing)
        else:
            first_thumb = f"/{run['first_frame']}" if run.get('first_frame') else ""
            last_thumb = f"/{run['last_frame']}" if run.get('last_frame') else ""
            
            cards_html += f'''
        <div class="card" data-folder="{run['folder']}" onclick="openModal('{run['folder']}')">
            <div class="card-thumbnails">
                <img src="{first_thumb}" alt="First frame" class="thumb thumb-first" loading="lazy">
                <div class="thumb-arrow">→</div>
                <img src="{last_thumb}" alt="Last frame" class="thumb thumb-last" loading="lazy">
            </div>
            <div class="card-info">
                <div class="card-header">
                    <span class="model-name">{run['model_short']}</span>
                    <span class="mode-badge">{run['mode']}</span>
                </div>
                <div class="card-meta">
                    <span class="date">{run['date']}</span>
                    <span class="frames">{run['total_frames']} frames</span>
                    <span class="cost">{run['total_cost']}</span>
                </div>
                <div class="card-status {status_class}">{run['status']}{f" ({run['progress']})" if run.get('progress') else ''}</div>
            </div>
        </div>
        '''
    
    # Embed runs data as JSON for the modal
    runs_json = json.dumps(runs, indent=2)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Loop Gallery</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-deep: #0a0a0f;
            --bg-card: #12121a;
            --bg-card-hover: #1a1a25;
            --bg-modal: #0d0d14;
            --accent: #f97316;
            --accent-dim: #c2410c;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border: #1e293b;
            --success: #22c55e;
            --warning: #eab308;
            --shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'DM Sans', system-ui, sans-serif;
            background: var(--bg-deep);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.5;
        }}
        
        /* Subtle grid background */
        body::before {{
            content: '';
            position: fixed;
            inset: 0;
            background-image: 
                linear-gradient(rgba(249, 115, 22, 0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(249, 115, 22, 0.02) 1px, transparent 1px);
            background-size: 48px 48px;
            pointer-events: none;
            z-index: 0;
        }}
        
        .container {{
            position: relative;
            z-index: 1;
            max-width: 1600px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        header {{
            margin-bottom: 3rem;
            padding-bottom: 2rem;
            border-bottom: 1px solid var(--border);
        }}
        
        h1 {{
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
        }}
        
        h1 span {{
            color: var(--accent);
        }}
        
        .subtitle {{
            color: var(--text-secondary);
            font-size: 1rem;
        }}
        
        .filters {{
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
        }}
        
        .filters select {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-family: inherit;
            font-size: 0.9rem;
            cursor: pointer;
            min-width: 160px;
        }}
        
        .filters select:hover {{
            border-color: var(--accent-dim);
        }}
        
        .filters select:focus {{
            outline: none;
            border-color: var(--accent);
        }}
        
        .card.hidden {{
            display: none;
        }}
        
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 1.5rem;
        }}
        
        .card {{
            background: var(--bg-card);
            border-radius: 12px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.2s ease;
            border: 1px solid var(--border);
        }}
        
        .card:hover {{
            background: var(--bg-card-hover);
            transform: translateY(-2px);
            box-shadow: var(--shadow);
            border-color: var(--accent-dim);
        }}
        
        .card-thumbnails {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 1rem;
            background: rgba(0, 0, 0, 0.3);
        }}
        
        .thumb {{
            flex: 1;
            height: 120px;
            object-fit: cover;
            border-radius: 6px;
            background: var(--bg-deep);
        }}
        
        .thumb-arrow {{
            color: var(--text-muted);
            font-size: 1.5rem;
            flex-shrink: 0;
        }}
        
        .card-info {{
            padding: 1rem 1.25rem 1.25rem;
        }}
        
        .card-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
        }}
        
        .model-name {{
            font-weight: 600;
            font-size: 1.1rem;
            color: var(--text-primary);
        }}
        
        .mode-badge {{
            background: var(--accent);
            color: #000;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}
        
        .mode-badge.mode-promptloop {{
            background: linear-gradient(135deg, #8b5cf6, #6366f1);
            color: #fff;
        }}
        
        .card-meta {{
            display: flex;
            gap: 1rem;
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }}
        
        .card-meta span {{
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }}
        
        .card-prompt {{
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            line-height: 1.4;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        
        .thumb-single {{
            width: 100%;
            height: 120px;
        }}
        
        .card-status {{
            font-size: 0.75rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .status-complete {{
            color: var(--success);
        }}
        
        .status-partial {{
            color: var(--warning);
        }}
        
        .status-progress {{
            color: #3b82f6;
        }}
        
        /* Modal */
        .modal-overlay {{
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.9);
            z-index: 1000;
            overflow-y: auto;
        }}
        
        .modal-overlay.active {{
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding: 2rem;
        }}
        
        .modal {{
            background: var(--bg-modal);
            border-radius: 16px;
            max-width: 1200px;
            width: 100%;
            border: 1px solid var(--border);
            box-shadow: 0 8px 48px rgba(0, 0, 0, 0.6);
            animation: modalIn 0.2s ease;
        }}
        
        @keyframes modalIn {{
            from {{
                opacity: 0;
                transform: scale(0.95);
            }}
            to {{
                opacity: 1;
                transform: scale(1);
            }}
        }}
        
        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.5rem 2rem;
            border-bottom: 1px solid var(--border);
        }}
        
        .modal-title {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        
        .modal-title h2 {{
            font-size: 1.5rem;
            font-weight: 600;
        }}
        
        .close-btn {{
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 2rem;
            cursor: pointer;
            padding: 0.5rem;
            line-height: 1;
            transition: color 0.2s;
        }}
        
        .close-btn:hover {{
            color: var(--text-primary);
        }}
        
        .modal-content {{
            padding: 2rem;
        }}
        
        .viewer {{
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }}
        
        .frame-display {{
            position: relative;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            aspect-ratio: 4/3;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .frame-display img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
        
        .frame-counter {{
            position: absolute;
            top: 1rem;
            right: 1rem;
            background: rgba(0, 0, 0, 0.7);
            padding: 0.4rem 0.8rem;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.875rem;
        }}
        
        .controls {{
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
            background: var(--bg-card);
            border-radius: 8px;
        }}
        
        .play-btn {{
            background: var(--accent);
            border: none;
            color: #000;
            width: 48px;
            height: 48px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            transition: transform 0.1s;
        }}
        
        .play-btn:hover {{
            transform: scale(1.05);
        }}
        
        .play-btn:active {{
            transform: scale(0.95);
        }}
        
        .nav-btn {{
            background: var(--bg-deep);
            border: 1px solid var(--border);
            color: var(--text-primary);
            width: 40px;
            height: 40px;
            border-radius: 6px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            transition: all 0.15s;
        }}
        
        .nav-btn:hover {{
            background: var(--bg-card-hover);
            border-color: var(--accent-dim);
        }}
        
        .nav-btn:disabled {{
            opacity: 0.3;
            cursor: not-allowed;
        }}
        
        .timeline {{
            flex: 1;
            height: 8px;
            background: var(--bg-deep);
            border-radius: 4px;
            cursor: pointer;
            position: relative;
        }}
        
        .timeline-progress {{
            height: 100%;
            background: var(--accent);
            border-radius: 4px;
            transition: width 0.1s;
        }}
        
        .speed-control {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--text-secondary);
            font-size: 0.875rem;
        }}
        
        .speed-control select {{
            background: var(--bg-deep);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.4rem 0.6rem;
            border-radius: 4px;
            font-family: inherit;
            cursor: pointer;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }}
        
        .stat-card {{
            background: var(--bg-card);
            padding: 1rem 1.25rem;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        
        .stat-label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }}
        
        .stat-value {{
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        .stat-value.mono {{
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .prompt-section {{
            margin-top: 1.5rem;
            padding: 1.25rem;
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        
        .prompt-label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .copy-btn {{
            background: none;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            padding: 0.25rem;
            display: flex;
            align-items: center;
            opacity: 0.6;
            transition: opacity 0.2s, color 0.2s;
        }}
        
        .copy-btn:hover {{
            opacity: 1;
            color: var(--text-secondary);
        }}
        
        .copy-btn:active {{
            color: var(--accent);
        }}
        
        .prompt-text {{
            color: var(--text-secondary);
            font-style: italic;
            line-height: 1.6;
            user-select: text;
            cursor: text;
        }}
        
        .path-section {{
            margin-top: 1rem;
            padding: 1rem 1.25rem;
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        
        .path-text {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--text-secondary);
            word-break: break-all;
            user-select: text;
            cursor: text;
        }}
        
        .text2img-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        
        .text2img-item {{
            background: var(--bg-card);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
            transition: transform 0.2s, border-color 0.2s;
        }}
        
        .text2img-item:hover {{
            transform: translateY(-2px);
            border-color: var(--accent-dim);
        }}
        
        .text2img-item img {{
            width: 100%;
            height: auto;
            display: block;
            cursor: pointer;
        }}
        
        .text2img-item-info {{
            padding: 1rem;
        }}
        
        .text2img-item-model {{
            font-weight: 600;
            font-size: 1rem;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }}
        
        .text2img-item-meta {{
            display: flex;
            gap: 1rem;
            font-size: 0.875rem;
            color: var(--text-secondary);
        }}
        
        .description-section {{
            margin-top: 1rem;
            padding: 1.25rem;
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid var(--border);
            border-left: 3px solid #8b5cf6;
        }}
        
        .description-section.hidden {{
            display: none;
        }}
        
        .description-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }}
        
        .description-label {{
            font-size: 0.75rem;
            color: #8b5cf6;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }}
        
        .description-frame {{
            font-size: 0.75rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .description-text {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            line-height: 1.6;
            max-height: 200px;
            overflow-y: auto;
            white-space: pre-wrap;
        }}
        
        .empty-state {{
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-muted);
        }}
        
        .empty-state h2 {{
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
            color: var(--text-secondary);
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 1rem;
            }}
            
            .gallery {{
                grid-template-columns: 1fr;
            }}
            
            .modal-overlay.active {{
                padding: 1rem;
            }}
            
            .modal-content {{
                padding: 1rem;
            }}
            
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Image <span>Loop</span> Gallery</h1>
            <p class="subtitle"><span id="visible-count">{len(runs)}</span> runs</p>
            <div class="filters">
                <select id="filter-model" onchange="applyFilters()">
                    <option value="">All Models</option>
                </select>
                <select id="filter-mode" onchange="applyFilters()">
                    <option value="">All Modes</option>
                </select>
            </div>
        </header>
        
        <div class="gallery">
            {cards_html if cards_html else '<div class="empty-state"><h2>No runs found</h2><p>Generate some image loops to see them here.</p></div>'}
        </div>
    </div>
    
    <div class="modal-overlay" id="modal" onclick="closeModalOnOverlay(event)">
        <div class="modal" onclick="event.stopPropagation()">
            <div class="modal-header">
                <div class="modal-title">
                    <h2 id="modal-model">Model</h2>
                    <span class="mode-badge" id="modal-mode">mode</span>
                </div>
                <button class="close-btn" onclick="closeModal()">&times;</button>
            </div>
            <div class="modal-content">
                <div class="viewer" id="viewer">
                    <div class="frame-display" id="frame-display">
                        <img id="frame-image" src="" alt="Frame">
                        <div class="frame-counter">
                            <span id="current-frame">1</span> / <span id="total-frames">20</span>
                        </div>
                    </div>
                    
                    <div class="controls" id="controls">
                        <button class="nav-btn" id="prev-btn" onclick="prevFrame()">◀</button>
                        <button class="play-btn" id="play-btn" onclick="togglePlay()">▶</button>
                        <button class="nav-btn" id="next-btn" onclick="nextFrame()">▶</button>
                        
                        <div class="timeline" id="timeline" onclick="seekTimeline(event)">
                            <div class="timeline-progress" id="timeline-progress"></div>
                        </div>
                        
                        <div class="speed-control">
                            <label>Speed:</label>
                            <select id="speed-select" onchange="changeSpeed()">
                                <option value="2000">0.5x</option>
                                <option value="1000" selected>1x</option>
                                <option value="500">2x</option>
                                <option value="250">4x</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="text2img-grid" id="text2img-grid" style="display: none;">
                    <!-- Text-to-image grid populated by JS -->
                </div>
                
                <div class="description-section hidden" id="description-section">
                    <div class="description-header">
                        <span class="description-label">Frame Description (image → text)</span>
                        <span class="description-frame" id="description-frame">Frame 1</span>
                    </div>
                    <div class="description-text" id="description-text">Loading...</div>
                </div>
                
                <div class="stats-grid" id="stats-grid">
                    <!-- Stats populated by JS -->
                </div>
                
                <div class="prompt-section">
                    <div class="prompt-label">
                        Prompt
                        <button class="copy-btn" onclick="copyToClipboard('modal-prompt')" title="Copy prompt (Ctrl+C)">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                            </svg>
                        </button>
                    </div>
                    <div class="prompt-text" id="modal-prompt">Loading...</div>
                </div>
                
                <div class="path-section">
                    <div class="prompt-label">
                        Path
                        <button class="copy-btn" onclick="copyToClipboard('modal-path')" title="Copy path (Ctrl+C)">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                            </svg>
                        </button>
                    </div>
                    <code class="path-text" id="modal-path">Loading...</code>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const runsData = {runs_json};
        
        // Populate filter dropdowns
        (function initFilters() {{
            const models = [...new Set(runsData.map(r => r.model_short))].sort();
            const modes = [...new Set(runsData.map(r => r.mode))].sort();
            
            const modelSelect = document.getElementById('filter-model');
            const modeSelect = document.getElementById('filter-mode');
            
            models.forEach(m => {{
                const opt = document.createElement('option');
                opt.value = m;
                opt.textContent = m;
                modelSelect.appendChild(opt);
            }});
            
            modes.forEach(m => {{
                const opt = document.createElement('option');
                opt.value = m;
                opt.textContent = m;
                modeSelect.appendChild(opt);
            }});
        }})();
        
        function applyFilters() {{
            const modelFilter = document.getElementById('filter-model').value;
            const modeFilter = document.getElementById('filter-mode').value;
            
            let visibleCount = 0;
            
            document.querySelectorAll('.card').forEach(card => {{
                const folder = card.dataset.folder;
                const run = runsData.find(r => r.folder === folder);
                if (!run) return;
                
                const matchModel = !modelFilter || run.model_short === modelFilter;
                const matchMode = !modeFilter || run.mode === modeFilter;
                
                if (matchModel && matchMode) {{
                    card.classList.remove('hidden');
                    visibleCount++;
                }} else {{
                    card.classList.add('hidden');
                }}
            }});
            
            document.getElementById('visible-count').textContent = visibleCount;
        }}
        
        let currentRun = null;
        let currentFrameIndex = 0;
        let isPlaying = false;
        let playInterval = null;
        let playSpeed = 1000;
        
        function openModal(folder) {{
            currentRun = runsData.find(r => r.folder === folder);
            if (!currentRun) return;
            
            const isText2Img = currentRun.type === 'text-to-image';
            
            // Show/hide appropriate viewer
            document.getElementById('viewer').style.display = isText2Img ? 'none' : 'flex';
            document.getElementById('text2img-grid').style.display = isText2Img ? 'grid' : 'none';
            
            currentFrameIndex = 0;
            
            // Update header
            document.getElementById('modal-model').textContent = currentRun.model_short;
            document.getElementById('modal-mode').textContent = currentRun.mode;
            
            // Update prompt
            document.getElementById('modal-prompt').textContent = currentRun.prompt || 'No prompt specified';
            
            // Update path
            document.getElementById('modal-path').textContent = '{output_dir}/' + currentRun.folder;
            
            const isPromptLoop = currentRun.type === 'prompt-loop';
            
            // Show/hide description section for prompt-loop
            const descSection = document.getElementById('description-section');
            descSection.classList.toggle('hidden', !isPromptLoop);
            
            if (isText2Img) {{
                // Text-to-image display
                const grid = document.getElementById('text2img-grid');
                const images = currentRun.images || [];
                
                // Reset navigation index
                currentText2ImgIndex = 0;
                
                grid.innerHTML = images.map((img, index) => {{
                    const timeStr = img.duration ? img.duration.toFixed(1) + 's' : '—';
                    const costStr = '$' + img.cost.toFixed(4);
                    return `
                    <div class="text2img-item">
                        <img src="/${{img.file}}" alt="${{img.model_short}}" loading="lazy" onclick="openImageInNewTab('/${{img.file}}')">
                        <div class="text2img-item-info">
                            <div class="text2img-item-model">${{img.model_short}}</div>
                            <div class="text2img-item-meta">
                                <span>Cost: ${{costStr}}</span>
                                <span>Time: ${{timeStr}}</span>
                            </div>
                        </div>
                    </div>
                `;
                }}).join('');
                
                // Store images for keyboard navigation
                window.text2imgImages = images;
                
                // Update stats
                const statsGrid = document.getElementById('stats-grid');
                statsGrid.innerHTML = `
                    <div class="stat-card">
                        <div class="stat-label">Total Cost</div>
                        <div class="stat-value mono">${{currentRun.total_cost}}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Total Time</div>
                        <div class="stat-value mono">${{currentRun.total_time}}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Models</div>
                        <div class="stat-value">${{currentRun.total_frames}}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Status</div>
                        <div class="stat-value">${{currentRun.status}}</div>
                    </div>
                `;
            }} else if (isPromptLoop) {{
                // Prompt loop display - similar to image loop but with description panel
                const statsGrid = document.getElementById('stats-grid');
                const dims = currentRun.dimensions;
                const dimsStr = dims ? `${{dims.width}}×${{dims.height}}` : 'Unknown';
                const describeMode = currentRun.describe_mode || 'detailed';
                
                statsGrid.innerHTML = `
                    <div class="stat-card">
                        <div class="stat-label">Total Cost</div>
                        <div class="stat-value mono">${{currentRun.total_cost}}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Total Time</div>
                        <div class="stat-value mono">${{currentRun.total_time}}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Frames</div>
                        <div class="stat-value">${{currentRun.total_frames}}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Dimensions</div>
                        <div class="stat-value mono">${{dimsStr}}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Describe Mode</div>
                        <div class="stat-value">${{describeMode}}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Status</div>
                        <div class="stat-value">${{currentRun.status}}</div>
                    </div>
                `;
                
                // Update total frames
                document.getElementById('total-frames').textContent = currentRun.frames.length;
                
                // Show first frame and description
                updateFrame();
                updateDescription();
            }} else {{
                // Image loop display (existing)
                const statsGrid = document.getElementById('stats-grid');
                const dims = currentRun.dimensions;
                const dimsStr = dims ? `${{dims.width}}×${{dims.height}}` : 'Unknown';
                
                statsGrid.innerHTML = `
                    <div class="stat-card">
                        <div class="stat-label">Total Cost</div>
                        <div class="stat-value mono">${{currentRun.total_cost}}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Total Time</div>
                        <div class="stat-value mono">${{currentRun.total_time}}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Frames</div>
                        <div class="stat-value">${{currentRun.total_frames}}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Dimensions</div>
                        <div class="stat-value mono">${{dimsStr}}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Size Preset</div>
                        <div class="stat-value">${{currentRun.size || 'auto'}}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Status</div>
                        <div class="stat-value">${{currentRun.status}}</div>
                    </div>
                `;
                
                // Update total frames
                document.getElementById('total-frames').textContent = currentRun.frames.length;
                
                // Show first frame
                updateFrame();
            }}
            
            // Show modal
            document.getElementById('modal').classList.add('active');
            document.body.style.overflow = 'hidden';
        }}
        
        function closeModal() {{
            stopPlay();
            document.getElementById('modal').classList.remove('active');
            document.body.style.overflow = '';
            currentRun = null;
        }}
        
        function closeModalOnOverlay(event) {{
            if (event.target.id === 'modal') {{
                closeModal();
            }}
        }}
        
        function updateFrame() {{
            if (!currentRun || !currentRun.frames.length) return;
            
            const frame = currentRun.frames[currentFrameIndex];
            const imgEl = document.getElementById('frame-image');
            imgEl.src = '/' + frame.file;
            
            document.getElementById('current-frame').textContent = currentFrameIndex + 1;
            
            // Update timeline
            const progress = ((currentFrameIndex + 1) / currentRun.frames.length) * 100;
            document.getElementById('timeline-progress').style.width = progress + '%';
            
            // Update nav buttons
            document.getElementById('prev-btn').disabled = currentFrameIndex === 0;
            document.getElementById('next-btn').disabled = currentFrameIndex >= currentRun.frames.length - 1;
            
            // Update description for prompt-loop
            if (currentRun.type === 'prompt-loop') {{
                updateDescription();
            }}
        }}
        
        function updateDescription() {{
            if (!currentRun || currentRun.type !== 'prompt-loop') return;
            
            const frame = currentRun.frames[currentFrameIndex];
            const descSection = document.getElementById('description-section');
            const descText = document.getElementById('description-text');
            const descFrame = document.getElementById('description-frame');
            
            // Frame 0 is the input, no description
            if (frame.number === 0 || !frame.description) {{
                descText.textContent = '(Input frame - no description)';
                descFrame.textContent = 'Frame ' + (currentFrameIndex + 1);
            }} else {{
                descText.textContent = frame.description;
                descFrame.textContent = 'Frame ' + frame.number + ' description';
            }}
        }}
        
        function nextFrame() {{
            if (!currentRun) return;
            if (currentFrameIndex < currentRun.frames.length - 1) {{
                currentFrameIndex++;
                updateFrame();
            }} else if (isPlaying) {{
                // Loop back to start when playing
                currentFrameIndex = 0;
                updateFrame();
            }}
        }}
        
        function prevFrame() {{
            if (!currentRun) return;
            if (currentFrameIndex > 0) {{
                currentFrameIndex--;
                updateFrame();
            }}
        }}
        
        function togglePlay() {{
            if (isPlaying) {{
                stopPlay();
            }} else {{
                startPlay();
            }}
        }}
        
        function startPlay() {{
            isPlaying = true;
            document.getElementById('play-btn').textContent = '⏸';
            playInterval = setInterval(nextFrame, playSpeed);
        }}
        
        function stopPlay() {{
            isPlaying = false;
            document.getElementById('play-btn').textContent = '▶';
            if (playInterval) {{
                clearInterval(playInterval);
                playInterval = null;
            }}
        }}
        
        function changeSpeed() {{
            playSpeed = parseInt(document.getElementById('speed-select').value);
            if (isPlaying) {{
                stopPlay();
                startPlay();
            }}
        }}
        
        function seekTimeline(event) {{
            if (!currentRun) return;
            const timeline = document.getElementById('timeline');
            const rect = timeline.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const percent = x / rect.width;
            currentFrameIndex = Math.floor(percent * currentRun.frames.length);
            currentFrameIndex = Math.max(0, Math.min(currentFrameIndex, currentRun.frames.length - 1));
            updateFrame();
        }}
        
        function copyToClipboard(elementId) {{
            const element = document.getElementById(elementId);
            const text = element.textContent || element.innerText;
            
            const copyWithFeedback = (btn) => {{
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard.writeText(text).then(() => {{
                        // Visual feedback
                        const originalColor = btn.style.color;
                        btn.style.color = 'var(--success)';
                        setTimeout(() => {{
                            btn.style.color = originalColor;
                        }}, 500);
                    }}).catch(err => {{
                        console.error('Failed to copy:', err);
                    }});
                }} else {{
                    // Fallback for older browsers
                    const textarea = document.createElement('textarea');
                    textarea.value = text;
                    textarea.style.position = 'fixed';
                    textarea.style.opacity = '0';
                    document.body.appendChild(textarea);
                    textarea.select();
                    try {{
                        document.execCommand('copy');
                        const originalColor = btn.style.color;
                        btn.style.color = 'var(--success)';
                        setTimeout(() => {{
                            btn.style.color = originalColor;
                        }}, 500);
                    }} catch (err) {{
                        console.error('Failed to copy:', err);
                    }}
                    document.body.removeChild(textarea);
                }}
            }};
            
            // Get the button that triggered this (from onclick) or find it
            const btn = event ? event.target.closest('.copy-btn') : 
                       document.querySelector(`.copy-btn[onclick*="${{elementId}}"]`);
            if (btn) {{
                copyWithFeedback(btn);
            }} else {{
                // Fallback: just copy without visual feedback
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard.writeText(text);
                }}
            }}
        }}
        
        function openImageInNewTab(imagePath) {{
            window.open(imagePath, '_blank');
        }}
        
        let currentText2ImgIndex = 0;
        
        function navigateText2Img(direction) {{
            if (!window.text2imgImages || window.text2imgImages.length === 0) return;
            
            if (direction === 'next') {{
                currentText2ImgIndex = (currentText2ImgIndex + 1) % window.text2imgImages.length;
            }} else {{
                currentText2ImgIndex = (currentText2ImgIndex - 1 + window.text2imgImages.length) % window.text2imgImages.length;
            }}
            
            const img = window.text2imgImages[currentText2ImgIndex];
            if (img && img.file) {{
                openImageInNewTab('/' + img.file);
            }}
        }}
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => {{
            // Always allow ESC to close modal
            if (e.key === 'Escape') {{
                if (document.getElementById('modal').classList.contains('active')) {{
                    closeModal();
                }}
                return;
            }}
            
            // Only handle other shortcuts when modal is open
            if (!currentRun || !document.getElementById('modal').classList.contains('active')) return;
            
            const isText2Img = currentRun.type === 'text-to-image';
            
            // Handle Ctrl+C for copying
            if (e.ctrlKey && e.key === 'c' && !e.shiftKey && !e.altKey) {{
                const activeElement = document.activeElement;
                if (activeElement && (activeElement.id === 'modal-prompt' || activeElement.id === 'modal-path')) {{
                    // Let default copy behavior work
                    return;
                }}
                // Try to copy prompt or path if focused
                if (document.getElementById('modal-prompt').contains(activeElement)) {{
                    e.preventDefault();
                    copyToClipboard('modal-prompt');
                    return;
                }}
                if (document.getElementById('modal-path').contains(activeElement)) {{
                    e.preventDefault();
                    copyToClipboard('modal-path');
                    return;
                }}
            }}
            
            switch(e.key) {{
                case 'ArrowLeft':
                    if (isText2Img) {{
                        navigateText2Img('prev');
                    }} else {{
                        prevFrame();
                    }}
                    break;
                case 'ArrowRight':
                    if (isText2Img) {{
                        navigateText2Img('next');
                    }} else {{
                        nextFrame();
                    }}
                    break;
                case ' ':
                    if (!isText2Img) {{
                        e.preventDefault();
                        togglePlay();
                    }}
                    break;
            }}
        }});
    </script>
</body>
</html>
'''
    return html


class GalleryHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for the gallery."""
    
    def __init__(self, *args, output_dir=None, **kwargs):
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        
        if path == "/" or path == "/index.html":
            # Serve the gallery page
            runs = get_runs(self.output_dir)
            html = generate_html(runs, self.output_dir)
            
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(html.encode()))
            self.end_headers()
            self.wfile.write(html.encode())
            return
        
        # Serve static files from output directory
        if path.startswith("/output/"):
            file_path = self.output_dir.parent / path[1:]  # Remove leading /
        else:
            file_path = self.output_dir.parent / path[1:]
        
        if file_path.exists() and file_path.is_file():
            # Determine content type
            content_type, _ = mimetypes.guess_type(str(file_path))
            if content_type is None:
                content_type = "application/octet-stream"
            
            with open(file_path, "rb") as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(content))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404, f"File not found: {path}")
    
    def log_message(self, format, *args):
        """Suppress default logging for cleaner output."""
        pass


def run_server(port: int = 8080, output_dir: Path = DEFAULT_OUTPUT_DIR):
    """Start the gallery web server."""
    
    # Create handler with output_dir bound
    def handler(*args, **kwargs):
        return GalleryHandler(*args, output_dir=output_dir, **kwargs)
    
    server = HTTPServer(("", port), handler)
    
    print(f"\n  🖼️  Image Loop Gallery")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  📂 Output dir: {output_dir}")
    print(f"  🌐 Server:     http://localhost:{port}")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"\n  Press Ctrl+C to stop\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Web gallery for viewing image loop runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8080,
        help="Port to run the server on (default: 8080)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory to scan for runs (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    if not args.output_dir.exists():
        print(f"Error: Output directory does not exist: {args.output_dir}")
        sys.exit(1)
    
    run_server(port=args.port, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
