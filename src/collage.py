# /// script
# dependencies = [
#   "Pillow",
# ]
# ///

"""
Collage Generator - Create a grid collage from image loop frames

Run with: uv run src/collage.py --help
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

# Output sizes
SIZES = {
    "small": (800, 600),
    "medium": (1600, 1200),
    "large": (3200, 2400),
}


def get_frame_files(folder: Path) -> list[Path]:
    """Get sorted list of frame files from a folder."""
    # Handle if user passes the run directory or images subdirectory
    if folder.name != "images" and (folder / "images").exists():
        folder = folder / "images"
    
    if not folder.exists():
        return []
    
    frames = sorted(folder.glob("frame_*.png"))
    return frames


def select_frames(frames: list[Path], count: int) -> list[Path]:
    """
    Select evenly distributed frames, always including first and last.
    
    Args:
        frames: List of all frame paths
        count: Number of frames to select
    
    Returns:
        Selected frame paths
    """
    if len(frames) < count:
        raise ValueError(f"Not enough frames: need {count}, have {len(frames)}")
    
    if count == 1:
        return [frames[0]]
    
    if count == 2:
        return [frames[0], frames[-1]]
    
    # Always include first and last, distribute the rest evenly
    selected = [frames[0]]
    
    # Calculate step for middle frames
    # We need to pick (count - 2) frames from the middle
    remaining = count - 2
    step = (len(frames) - 1) / (count - 1)
    
    for i in range(1, count - 1):
        idx = int(i * step)
        selected.append(frames[idx])
    
    selected.append(frames[-1])
    return selected


def create_collage(
    frames: list[Path],
    grid: tuple[int, int],
    output_size: tuple[int, int],
    output_path: Path,
) -> None:
    """
    Create a grid collage from frames.
    
    Args:
        frames: List of frame paths to include
        grid: (columns, rows) tuple
        output_size: (width, height) of output image
        output_path: Where to save the collage
    """
    cols, rows = grid
    total_w, total_h = output_size
    
    # Calculate cell size
    cell_w = total_w // cols
    cell_h = total_h // rows
    
    # Create output image
    collage = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    
    for i, frame_path in enumerate(frames):
        row = i // cols
        col = i % cols
        
        # Load and resize image to fit cell (crop to fill)
        img = Image.open(frame_path)
        
        # Calculate scaling to cover cell
        img_w, img_h = img.size
        scale = max(cell_w / img_w, cell_h / img_h)
        
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Center crop to cell size
        left = (new_w - cell_w) // 2
        top = (new_h - cell_h) // 2
        img = img.crop((left, top, left + cell_w, top + cell_h))
        
        # Paste into collage
        x = col * cell_w
        y = row * cell_h
        collage.paste(img, (x, y))
    
    # Save
    collage.save(output_path, "PNG")
    print(f"✅ Collage saved: {output_path}")
    print(f"   Grid: {cols}x{rows} ({cols * rows} images)")
    print(f"   Size: {total_w}x{total_h}")


def parse_grid(grid_str: str) -> tuple[int, int]:
    """Parse grid string like '3x3' into (cols, rows) tuple."""
    try:
        parts = grid_str.lower().split("x")
        if len(parts) != 2:
            raise ValueError()
        cols, rows = int(parts[0]), int(parts[1])
        if cols < 1 or rows < 1:
            raise ValueError()
        return cols, rows
    except ValueError:
        raise ValueError(f"Invalid grid format: '{grid_str}'. Use format like '3x3' or '4x2'")


def main():
    parser = argparse.ArgumentParser(
        description="Create a grid collage from image loop frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a 3x3 collage from a run
  uv run src/collage.py output/run_flux-pro_evolve_1218_1234_abcd --grid 3x3

  # Create a large 5x4 collage
  uv run src/collage.py output/run_flux-pro_evolve_1218_1234_abcd --grid 5x4 --size large

  # Specify output path
  uv run src/collage.py output/run_flux-pro_evolve_1218_1234_abcd --grid 4x4 -o my_collage.png
"""
    )
    
    parser.add_argument(
        "folder",
        help="Path to run folder or images directory",
    )
    parser.add_argument(
        "--grid", "-g",
        default="3x3",
        help="Grid size as COLSxROWS (default: 3x3)",
    )
    parser.add_argument(
        "--size", "-s",
        choices=list(SIZES.keys()),
        default="medium",
        help=f"Output size: {', '.join(f'{k} {v}' for k, v in SIZES.items())} (default: medium)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: collage.png in the run folder)",
    )
    
    args = parser.parse_args()
    
    # Parse grid
    try:
        grid = parse_grid(args.grid)
    except ValueError as e:
        print(f"❌ {e}")
        return 1
    
    cols, rows = grid
    total_cells = cols * rows
    
    # Get frames
    folder = Path(args.folder)
    frames = get_frame_files(folder)
    
    if not frames:
        print(f"❌ No frames found in {folder}")
        return 1
    
    print(f"📁 Found {len(frames)} frames in {folder}")
    
    # Check we have enough
    if len(frames) < total_cells:
        print(f"❌ Not enough frames: need {total_cells} for {cols}x{rows} grid, have {len(frames)}")
        return 1
    
    # Select frames
    selected = select_frames(frames, total_cells)
    print(f"📸 Selected {len(selected)} frames (first: {selected[0].name}, last: {selected[-1].name})")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Save in run folder
        run_folder = folder if folder.name != "images" else folder.parent
        output_path = run_folder / f"collage_{cols}x{rows}.png"
    
    # Create collage
    output_size = SIZES[args.size]
    create_collage(selected, grid, output_size, output_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

