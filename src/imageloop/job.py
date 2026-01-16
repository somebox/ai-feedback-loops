"""Job orchestration - frame management and output generation."""

import glob
from pathlib import Path

from PIL import Image


def find_last_frame(images_dir: Path) -> tuple[int, Path] | None:
    """Find the last frame in a directory. Returns (frame_number, path) or None."""
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


def generate_video(images_dir: Path, output_file: Path, frame_rate: int = 12):
    """Generate an MP4 video from a directory of frame images."""
    try:
        import ffmpeg
    except ImportError:
        print("⚠️  ffmpeg-python not available. Install with: pip install ffmpeg-python")
        return False
    
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


def generate_gif(images_dir: Path, output_file: Path, frame_rate: int = 1):
    """Generate an animated GIF from a directory of frame images."""
    frame_pattern = str(images_dir / "frame_*.png")
    frame_files = sorted(glob.glob(frame_pattern))
    num_frames = len(frame_files)

    if num_frames < 2:
        print(f"⚠️  Not enough frames ({num_frames}) to generate GIF")
        return False

    print(f"\n🎞️  Generating GIF from {num_frames} frames...")

    try:
        # Load all frames
        frames = [Image.open(f) for f in frame_files]
        
        # Calculate duration per frame in milliseconds
        duration_ms = int(1000 / frame_rate)
        
        # Save as animated GIF
        frames[0].save(
            output_file,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,  # 0 = infinite loop
            optimize=False,
        )
        
        # Close all frames
        for frame in frames:
            frame.close()

        file_size = output_file.stat().st_size / (1024 * 1024)
        print(f"✅ GIF saved: {output_file} ({file_size:.1f} MB)")
        return True

    except Exception as e:
        print(f"❌ GIF generation failed: {e}")
        return False


def generate_outputs(images_dir: Path, run_dir: Path, frame_rate: int = 1, output_format: str = "mp4"):
    """Generate output files (MP4, GIF, or both) from frame images."""
    if output_format in ("mp4", "both"):
        video_path = run_dir / "animation.mp4"
        generate_video(images_dir, video_path, frame_rate)
    
    if output_format in ("gif", "both"):
        gif_path = run_dir / "animation.gif"
        generate_gif(images_dir, gif_path, frame_rate)
