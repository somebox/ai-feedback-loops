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


def test_runlog_tracks_prompt_loop_fields(temp_output_dir):
    """RunLog tracks description and describe_usage for prompt loop mode."""
    from imageloop.runlog import RunLog
    
    run_dir = temp_output_dir / "test_run"
    run_dir.mkdir()
    log = RunLog(run_dir=run_dir)
    
    log.start_session(is_continuation=False)
    
    # Log a prompt loop frame with both describe and render usage
    log.log_frame(
        frame_number=1,
        success=True,
        description="A beautiful landscape with mountains and a lake.",
        description_file="descriptions/frame_001.txt",
        describe_usage={"cost": 0.001, "total_tokens": 50, "input_tokens": 40, "output_tokens": 10},
        describe_duration_seconds=1.5,
        usage={"cost": 0.02, "total_tokens": 200, "input_tokens": 150, "output_tokens": 50},
        file_path="images/frame_001.png",
        file_size_bytes=2048,
        duration_seconds=5.0,
    )
    
    log.end_session()
    
    # Check frame entry has description fields
    assert len(log.frames) == 1
    frame = log.frames[0]
    assert frame["description"] == "A beautiful landscape with mountains and a lake."
    assert frame["description_file"] == "descriptions/frame_001.txt"
    assert frame["describe_usage"]["cost"] == 0.001
    assert frame["describe_duration_seconds"] == 1.5
    
    # Check cumulative stats include both describe and render costs
    assert log.stats["total_cost"] == pytest.approx(0.021, rel=0.01)
    assert log.stats["api_calls"] == 2  # One for describe, one for render
    assert log.stats["total_tokens"] == 250  # 50 + 200


def test_runlog_config_has_describe_fields(temp_output_dir):
    """RunLog config includes describe_mode and describe_prompt."""
    from imageloop.runlog import RunLog
    
    run_dir = temp_output_dir / "test_run"
    run_dir.mkdir()
    log = RunLog(run_dir=run_dir)
    
    log.set_config(
        model="test-model",
        mode="prompt-loop",
        describe_mode="detailed",
        describe_prompt="Describe this image in detail.",
    )
    
    assert log.config["describe_mode"] == "detailed"
    assert log.config["describe_prompt"] == "Describe this image in detail."
    
    # Save and reload
    log_path = log.save()
    loaded = RunLog.load(log_path)
    
    assert loaded.config["describe_mode"] == "detailed"
    assert loaded.config["describe_prompt"] == "Describe this image in detail."
