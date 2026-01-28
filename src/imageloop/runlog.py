"""Run logging and JSON persistence."""

import json
import time
from datetime import datetime
from pathlib import Path


class RunLog:
    """
    Comprehensive run logging with JSON persistence.
    
    Tracks all generation details including per-frame stats, API responses,
    errors, and continuation history.
    """

    def __init__(self, run_dir: Path = None):
        self.run_dir = run_dir
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        
        # Run configuration
        self.config = {
            "input_image": None,
            "model": None,
            "mode": None,
            "prompt": None,
            "temperature": None,
            "top_p": None,
            "seed": None,
            "size": None,
            "frame_dimensions": None,
            "requested_frames": None,
            "fps": None,
            "output_format": None,
            "command_line": None,
            # Prompt loop specific
            "describe_mode": None,
            "describe_prompt": None,
        }
        
        # Cumulative stats (across all sessions)
        self.stats = {
            "total_frames": 0,
            "frames_generated": 0,
            "frames_failed": 0,
            "api_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_time_seconds": 0.0,
        }
        
        # Session tracking (for continuations)
        self.sessions = []
        self._current_session = None
        
        # Per-frame details
        self.frames = []

    def start_session(self, is_continuation: bool = False):
        """Start a new generation session."""
        self._session_start = time.time()
        self._current_session = {
            "started_at": datetime.now().isoformat(),
            "is_continuation": is_continuation,
            "starting_frame": len(self.frames),
            "frames_requested": 0,
            "frames_generated": 0,
            "frames_failed": 0,
            "api_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "duration_seconds": 0.0,
        }

    def end_session(self):
        """End the current session and record stats."""
        if self._current_session:
            self._current_session["ended_at"] = datetime.now().isoformat()
            self._current_session["duration_seconds"] = time.time() - self._session_start
            self.sessions.append(self._current_session)
            
            # Update cumulative stats
            self.stats["total_time_seconds"] += self._current_session["duration_seconds"]
            self.updated_at = datetime.now().isoformat()
            
            self._current_session = None

    def set_config(self, **kwargs):
        """Set run configuration values."""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value

    def log_frame(
        self,
        frame_number: int,
        success: bool,
        usage: dict = None,
        api_response: dict = None,
        file_path: str = None,
        file_size_bytes: int = None,
        output_dimensions: tuple = None,
        duration_seconds: float = None,
        error: str = None,
        model_text: str = None,
        # Prompt loop fields
        description: str = None,
        description_file: str = None,
        describe_usage: dict = None,
        describe_duration_seconds: float = None,
    ):
        """Log details for a single frame generation.
        
        For prompt-loop mode, each frame includes both the description step
        (image-to-text) and the render step (text-to-image).
        """
        frame_entry = {
            "frame_number": frame_number,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "duration_seconds": duration_seconds,
        }
        
        # Prompt loop: description from image-to-text step
        # Store full description - it's needed for the render step and gallery display
        if description is not None:
            frame_entry["description"] = description
        if description_file:
            frame_entry["description_file"] = description_file
        if describe_usage:
            frame_entry["describe_usage"] = {
                "input_tokens": describe_usage.get("input_tokens", 0) or describe_usage.get("prompt_tokens", 0),
                "output_tokens": describe_usage.get("output_tokens", 0) or describe_usage.get("completion_tokens", 0),
                "total_tokens": describe_usage.get("total_tokens", 0),
                "cost": describe_usage.get("cost", 0.0),
            }
            # Add describe usage to session and cumulative stats
            if self._current_session:
                self._current_session["input_tokens"] += frame_entry["describe_usage"]["input_tokens"]
                self._current_session["output_tokens"] += frame_entry["describe_usage"]["output_tokens"]
                self._current_session["total_tokens"] += frame_entry["describe_usage"]["total_tokens"]
                self._current_session["cost"] += frame_entry["describe_usage"]["cost"]
                self._current_session["api_calls"] += 1
            
            self.stats["input_tokens"] += frame_entry["describe_usage"]["input_tokens"]
            self.stats["output_tokens"] += frame_entry["describe_usage"]["output_tokens"]
            self.stats["total_tokens"] += frame_entry["describe_usage"]["total_tokens"]
            self.stats["total_cost"] += frame_entry["describe_usage"]["cost"]
            self.stats["api_calls"] += 1
        if describe_duration_seconds is not None:
            frame_entry["describe_duration_seconds"] = describe_duration_seconds
        
        if file_path:
            frame_entry["file"] = str(file_path)
        if file_size_bytes:
            frame_entry["file_size_bytes"] = file_size_bytes
        if output_dimensions:
            frame_entry["dimensions"] = {"width": output_dimensions[0], "height": output_dimensions[1]}
        
        # API usage details
        if usage:
            frame_entry["usage"] = {
                "input_tokens": usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0) or usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "cost": usage.get("cost", 0.0),
            }
            
            # Update session stats
            if self._current_session:
                self._current_session["input_tokens"] += frame_entry["usage"]["input_tokens"]
                self._current_session["output_tokens"] += frame_entry["usage"]["output_tokens"]
                self._current_session["total_tokens"] += frame_entry["usage"]["total_tokens"]
                self._current_session["cost"] += frame_entry["usage"]["cost"]
                self._current_session["api_calls"] += 1
            
            # Update cumulative stats
            self.stats["input_tokens"] += frame_entry["usage"]["input_tokens"]
            self.stats["output_tokens"] += frame_entry["usage"]["output_tokens"]
            self.stats["total_tokens"] += frame_entry["usage"]["total_tokens"]
            self.stats["total_cost"] += frame_entry["usage"]["cost"]
            self.stats["api_calls"] += 1
        
        # Store raw API response for debugging (optional, can be large)
        if api_response:
            # Store only key metadata, not the full response
            frame_entry["api_response"] = {
                "id": api_response.get("id"),
                "model": api_response.get("model"),
                "output_types": [item.get("type") for item in api_response.get("output", [])],
            }
            if api_response.get("error"):
                frame_entry["api_response"]["error"] = api_response.get("error")
        
        if error:
            frame_entry["error"] = error
        if model_text:
            # Truncate long model responses
            frame_entry["model_text"] = model_text[:500] if len(model_text) > 500 else model_text
        
        self.frames.append(frame_entry)
        
        # Update frame counts
        if success:
            self.stats["frames_generated"] += 1
            if self._current_session:
                self._current_session["frames_generated"] += 1
        else:
            self.stats["frames_failed"] += 1
            if self._current_session:
                self._current_session["frames_failed"] += 1
        
        self.stats["total_frames"] = len([f for f in self.frames if f["success"]])
        
        # Auto-save after each frame if run_dir is set
        if self.run_dir:
            self._auto_save()

    def _auto_save(self):
        """Save the run log automatically (called after each frame)."""
        try:
            path = self.run_dir / "run.json"
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            print(f"\n⚠️  Failed to auto-save run log: {e}")

    def mark_interrupted(self):
        """Mark the run as interrupted and save."""
        if self._current_session:
            self._current_session["ended_at"] = datetime.now().isoformat()
            self._current_session["duration_seconds"] = time.time() - self._session_start
            self._current_session["interrupted"] = True
            self.sessions.append(self._current_session)
            self.stats["total_time_seconds"] += self._current_session["duration_seconds"]
            self._current_session = None
        
        self.updated_at = datetime.now().isoformat()
        if self.run_dir:
            self._auto_save()

    def to_dict(self) -> dict:
        """Export the full run log as a dictionary."""
        # Calculate derived stats
        stats = self.stats.copy()
        if stats["frames_generated"] > 0:
            stats["avg_time_per_frame"] = stats["total_time_seconds"] / stats["frames_generated"]
            stats["avg_cost_per_frame"] = stats["total_cost"] / stats["frames_generated"]
        else:
            stats["avg_time_per_frame"] = 0
            stats["avg_cost_per_frame"] = 0
        
        # Determine status
        has_interrupted = any(s.get("interrupted") for s in self.sessions)
        if has_interrupted:
            status = "interrupted"
        elif stats["frames_failed"] > 0:
            status = "partial"
        else:
            status = "completed"
        
        return {
            # Summary at top for human readability
            "summary": {
                "created": self.created_at,
                "updated": self.updated_at,
                "model": self.config.get("model"),
                "mode": self.config.get("mode"),
                "total_frames": stats["total_frames"],
                "total_cost": f"${stats['total_cost']:.4f}",
                "total_time": f"{stats['total_time_seconds']:.1f}s",
                "status": status,
            },
            "config": self.config,
            "stats": stats,
            "sessions": self.sessions,
            "frames": self.frames,
        }

    def save(self, path: Path = None):
        """Save the run log to a JSON file."""
        if path is None:
            if self.run_dir:
                path = self.run_dir / "run.json"
            else:
                raise ValueError("No path specified and no run_dir set")
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return path

    @classmethod
    def load(cls, path: Path) -> "RunLog":
        """Load a run log from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        log = cls(run_dir=path.parent)
        log.created_at = data.get("summary", {}).get("created", log.created_at)
        log.updated_at = data.get("summary", {}).get("updated", log.updated_at)
        log.config = data.get("config", log.config)
        log.stats = data.get("stats", log.stats)
        log.sessions = data.get("sessions", [])
        log.frames = data.get("frames", [])
        
        return log

    def print_summary(self, show_continue_command: bool = True, continue_command: str = None):
        """Print a human-readable summary to console."""
        stats = self.stats
        elapsed = stats["total_time_seconds"]
        
        lines = [
            "",
            "=" * 50,
            "📊 Generation Report",
            "=" * 50,
            f"⏱️  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)",
            f"🖼️  Frames generated: {stats['frames_generated']}",
            f"❌ Frames failed: {stats['frames_failed']}",
            f"🔄 API calls: {stats['api_calls']}",
            f"📥 Input tokens: {stats['input_tokens']:,}",
            f"📤 Output tokens: {stats['output_tokens']:,}",
            f"📊 Total tokens: {stats['total_tokens']:,}",
            f"💰 Total cost: ${stats['total_cost']:.4f}",
        ]
        if stats["frames_generated"] > 0:
            lines.append(f"⚡ Avg time per frame: {elapsed/stats['frames_generated']:.1f}s")
            lines.append(f"💵 Cost per frame: ${stats['total_cost']/stats['frames_generated']:.4f}")
        lines.append("=" * 50)
        
        # Show original command line if available
        command_line = self.config.get("command_line")
        if command_line:
            lines.append("")
            lines.append("📋 Original command:")
            lines.append(f"   {command_line}")
            lines.append("")
        
        # Add continue command if run_dir is available
        if show_continue_command and self.run_dir:
            if continue_command:
                # Use provided continue command
                lines.append("")
                lines.append("🔄 To continue this run:")
                lines.append(f"   {continue_command}")
                lines.append("")
            else:
                # Build continue command
                run_path = self.run_dir
                # Use relative path if possible for cleaner output
                try:
                    run_path = run_path.relative_to(Path.cwd())
                except ValueError:
                    pass
                
                # Get default frames from config or use 10
                requested = self.config.get("requested_frames", 10)
                default_frames = requested if requested > 0 else 10
                
                lines.append("")
                lines.append("🔄 To continue this run:")
                lines.append(f"   python src/image_loop.py --continue {run_path} --frames {default_frames}")
                lines.append("")
        
        print("\n".join(lines))


def parse_legacy_report(report_path: Path) -> dict:
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
