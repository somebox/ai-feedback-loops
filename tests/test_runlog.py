"""Tests for imageloop.runlog module - run logging and persistence."""

import json
from pathlib import Path
import time
import pytest


def test_runlog_saves_and_loads(temp_output_dir):
    """RunLog round-trips through JSON correctly."""
    from imageloop.runlog import RunLog
    
    # Create and configure run log
    run_dir = temp_output_dir / "test_run"
    run_dir.mkdir()
    log = RunLog(run_dir=run_dir)
    
    log.set_config(
        model="test-model",
        mode="evolve",
        frames=5,
    )
    
    # Save
    log_path = log.save()
    assert log_path.exists()
    
    # Load
    loaded = RunLog.load(log_path)
    
    # Verify config loaded correctly
    assert loaded.config["model"] == "test-model"
    assert loaded.config["mode"] == "evolve"
    assert loaded.run_dir == run_dir


def test_runlog_tracks_frame_success(temp_output_dir):
    """Logging successful frame updates stats."""
    from imageloop.runlog import RunLog
    
    run_dir = temp_output_dir / "test_run"
    run_dir.mkdir()
    log = RunLog(run_dir=run_dir)
    
    # Log successful frame
    log.log_frame(
        frame_number=1,
        success=True,
        usage={"cost": 0.01, "total_tokens": 100},
        file_path="frame_001.png",
        file_size_bytes=1024,
    )
    
    assert log.stats["frames_generated"] == 1
    assert log.stats["frames_failed"] == 0
    assert log.stats["total_cost"] == 0.01
    assert log.stats["total_tokens"] == 100


def test_runlog_tracks_frame_failure(temp_output_dir):
    """Logging failed frame updates failure count."""
    from imageloop.runlog import RunLog
    
    run_dir = temp_output_dir / "test_run"
    run_dir.mkdir()
    log = RunLog(run_dir=run_dir)
    
    # Log failed frame
    log.log_frame(
        frame_number=1,
        success=False,
        error="API error",
        usage={"cost": 0.01},
    )
    
    assert log.stats["frames_generated"] == 0
    assert log.stats["frames_failed"] == 1
    # Cost is still tracked even for failures
    assert log.stats["total_cost"] == 0.01


def test_runlog_session_timing(temp_output_dir):
    """Session duration calculated on end_session."""
    from imageloop.runlog import RunLog
    
    run_dir = temp_output_dir / "test_run"
    run_dir.mkdir()
    log = RunLog(run_dir=run_dir)
    
    # Start session
    log.start_session(is_continuation=False)
    
    # Wait a bit
    time.sleep(0.1)
    
    # End session
    log.end_session()
    
    # Check duration was calculated
    assert len(log.sessions) == 1
    assert log.sessions[0]["duration_seconds"] >= 0.1
    assert "ended_at" in log.sessions[0]


def test_runlog_interrupted_status(temp_output_dir):
    """Interrupted runs marked in status."""
    from imageloop.runlog import RunLog
    
    run_dir = temp_output_dir / "test_run"
    run_dir.mkdir()
    log = RunLog(run_dir=run_dir)
    
    log.start_session(is_continuation=False)
    log.mark_interrupted()
    
    # Status should be "interrupted" in summary
    summary = log.to_dict()["summary"]
    assert summary["status"] == "interrupted"
    
    # Session should be marked as interrupted
    assert len(log.sessions) == 1
    assert log.sessions[0].get("interrupted") is True


def test_legacy_report_parsing(tmp_path):
    """Old report.txt files parse into settings dict."""
    from imageloop.runlog import parse_legacy_report
    
    # Create a legacy report file
    report_path = tmp_path / "report.txt"
    report_path.write_text("""Image Loop Generation Report
==================================================
Generated: 2025-01-15T10:00:00
Model: black-forest-labs/flux.2-pro
Mode: evolve
Prompt: Transform this image slightly
Temperature: 0.7
Top P: 0.9
Size: landscape (1024x768)
""")
    
    settings = parse_legacy_report(report_path)
    
    assert settings["model"] == "black-forest-labs/flux.2-pro"
    assert settings["mode"] == "evolve"
    assert settings["temperature"] == 0.7
    assert settings["top_p"] == 0.9
    assert settings["size"] == "landscape"


def test_runlog_auto_saves_after_frame(temp_output_dir):
    """RunLog auto-saves after logging each frame."""
    from imageloop.runlog import RunLog
    
    run_dir = temp_output_dir / "test_run"
    run_dir.mkdir()
    log = RunLog(run_dir=run_dir)
    
    # Log a frame (should auto-save)
    log.log_frame(
        frame_number=1,
        success=True,
        usage={"cost": 0.01},
    )
    
    # Check file was created
    log_file = run_dir / "run.json"
    assert log_file.exists()
    
    # Verify it can be loaded
    loaded = RunLog.load(log_file)
    assert len(loaded.frames) == 1
    assert loaded.frames[0]["frame_number"] == 1
