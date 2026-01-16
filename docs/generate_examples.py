# /// script
# dependencies = [
#   "pyyaml",
# ]
# ///

#!/usr/bin/env python3
"""
Generate examples.md from run directories listed in manifest.json.

This script reads the manifest.json file, loads run.json from each directory,
extracts the command line, cost, model, and mode, copies the animation.gif,
and generates an examples.md file.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imageloop import runlog, settings


def main():
    """Generate examples.md from manifest.json."""
    docs_dir = Path(__file__).parent
    manifest_path = docs_dir / "manifest.json"
    examples_dir = docs_dir / "examples"
    examples_md_path = docs_dir / "examples.md"
    output_dir = Path(__file__).parent.parent / "output"
    
    # Load manifest
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        return 1
    
    with open(manifest_path) as f:
        run_dirs = json.load(f)
    
    # Ensure examples directory exists
    examples_dir.mkdir(exist_ok=True)
    
    # Clean up old GIFs (those with long names containing "run_")
    old_gifs = list(examples_dir.glob("*run_*.gif"))
    for old_gif in old_gifs:
        try:
            old_gif.unlink()
            print(f"🗑️  Removed old GIF: {old_gif.name}")
        except Exception as e:
            print(f"⚠️  Failed to remove {old_gif}: {e}")
    
    # Load settings for model name resolution
    cfg = settings.load_settings()
    
    examples = []
    
    for run_dir_rel in run_dirs:
        run_dir = Path(__file__).parent.parent / run_dir_rel
        
        if not run_dir.exists():
            print(f"⚠️  Run directory not found: {run_dir}")
            continue
        
        run_json = run_dir / "run.json"
        animation_gif = run_dir / "animation.gif"
        
        if not run_json.exists():
            print(f"⚠️  Run JSON not found: {run_json}")
            continue
        
        if not animation_gif.exists():
            print(f"⚠️  Animation GIF not found: {animation_gif}")
            continue
        
        # Load run log
        try:
            run_log = runlog.RunLog.load(run_json)
        except Exception as e:
            print(f"⚠️  Failed to load {run_json}: {e}")
            continue
        
        # Extract information
        summary = run_log.to_dict().get("summary", {})
        config = run_log.config
        
        command_line = config.get("command_line", "unknown")
        cost = summary.get("total_cost", "$0.00")
        model_full = summary.get("model", config.get("model", "unknown"))
        model_short = settings.get_model_short_name(model_full, cfg)
        mode = summary.get("mode", config.get("mode", "unknown"))
        frames = summary.get("total_frames", config.get("requested_frames", 0))
        
        # Copy GIF to examples directory with a clean name
        # Extract unique ID from run directory name (last part after timestamp)
        run_id = run_dir.name.split('_')[-1] if '_' in run_dir.name else run_dir.name[-4:]
        gif_name = f"{model_short}_{mode}_{run_id}.gif"
        gif_dest = examples_dir / gif_name
        
        try:
            shutil.copy2(animation_gif, gif_dest)
            print(f"✅ Copied: {gif_name}")
            
            # Optimize GIF using gifsicle
            try:
                opt_gif = gif_dest.with_name(f"{gif_dest.stem}_opt.gif")
                result = subprocess.run(
                    ["gifsicle", "-O3", "--lossy=30", "--scale", "0.75", str(gif_dest), "-o", str(opt_gif)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0 and opt_gif.exists():
                    original_size = gif_dest.stat().st_size
                    opt_size = opt_gif.stat().st_size
                    # Replace original with optimized version
                    opt_gif.replace(gif_dest)
                    savings = original_size - opt_size
                    savings_pct = (savings / original_size * 100) if original_size > 0 else 0
                    print(f"   ✨ Optimized: {opt_size // 1024}KB (saved {savings // 1024}KB, {savings_pct:.1f}%)")
                else:
                    if result.stderr:
                        print(f"   ⚠️  gifsicle warning: {result.stderr.strip()}")
            except FileNotFoundError:
                print(f"   ⚠️  gifsicle not found - GIF not optimized (install with: brew install gifsicle)")
            except subprocess.TimeoutExpired:
                print(f"   ⚠️  gifsicle timeout - GIF not optimized")
            except Exception as e:
                print(f"   ⚠️  Optimization failed: {e}")
                
        except Exception as e:
            print(f"⚠️  Failed to copy {animation_gif}: {e}")
            continue
        
        # Store example data
        examples.append({
            "command": command_line,
            "cost": cost,
            "model": model_short,
            "model_full": model_full,
            "mode": mode,
            "frames": frames,
            "gif_path": f"examples/{gif_name}",
            "gif_name": gif_name,
        })
    
    # Generate markdown
    md_lines = [
        "# Examples",
        "",
        "Generated examples of image loop animations.",
        "",
    ]
    
    for i, example in enumerate(examples, 1):
        md_lines.extend([
            f"## Example {i}: {example['mode'].title()} with {example['model']}",
            "",
            f"**Command:**",
            "```bash",
            example['command'],
            "```",
            "",
            f"**Model:** {example['model']} ({example['model_full']})",
            f"**Cost:** {example['cost']}",
            f"**Frames:** {example['frames']}",
            "",
            f"![{example['mode']} animation]({example['gif_path']})",
            "",
            "---",
            "",
        ])
    
    # Write examples.md
    with open(examples_md_path, "w") as f:
        f.write("\n".join(md_lines))
    
    print(f"\n✅ Generated: {examples_md_path}")
    print(f"   Examples: {len(examples)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
