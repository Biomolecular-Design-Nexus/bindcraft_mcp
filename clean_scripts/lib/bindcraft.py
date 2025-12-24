"""Core BindCraft functionality extracted from examples/bindcraft_utils.py.

This module contains the essential BindCraft operations with minimal dependencies.
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List


# Configuration
SCRIPT_DIR = Path(__file__).parent
MCP_ROOT = SCRIPT_DIR.parent.parent

PATHS = {
    "bindcraft_scripts": MCP_ROOT / "scripts",
    "examples_data": MCP_ROOT / "examples" / "data",
    "configs": MCP_ROOT / "configs"
}

DEFAULT_CONFIG = {
    "device": "0",
    "timeout": 3600,  # 1 hour default timeout
}


def get_bindcraft_path() -> Path:
    """Get the BindCraft scripts path."""
    bindcraft_path = PATHS["bindcraft_scripts"]

    if not bindcraft_path.exists():
        raise FileNotFoundError(
            f"BindCraft scripts not found at {bindcraft_path}. "
            "Please ensure the scripts directory exists."
        )
    return bindcraft_path


def get_default_settings_paths() -> dict:
    """Get default paths for filter and advanced settings."""
    examples_data_path = PATHS["examples_data"]
    return {
        "filters": str(examples_data_path / "default_filters.json"),
        "advanced": str(examples_data_path / "default_4stage_multimer.json"),
    }


def log_stream(stream, logs: List[str], prefix: str = ""):
    """Collect output from a stream and print in real-time."""
    try:
        for line in iter(stream.readline, b''):
            line_str = line.decode('utf-8').rstrip()
            if line_str:
                output = f"{prefix}{line_str}" if prefix else line_str
                print(output, flush=True)
                logs.append(output)
    except Exception:
        pass
    finally:
        stream.close()


def run_command(
    cmd: List[str],
    device: str = "0",
    timeout: int = 3600,
    cwd: Optional[str] = None
) -> Dict[str, Any]:
    """Run a command and return results."""
    print(f"🚀 Running: {' '.join(cmd)}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)

    logs = []

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=cwd
        )

        # Start threads to log output
        stdout_thread = threading.Thread(
            target=log_stream,
            args=(process.stdout, logs, "")
        )
        stderr_thread = threading.Thread(
            target=log_stream,
            args=(process.stderr, logs, "ERROR: ")
        )

        stdout_thread.start()
        stderr_thread.start()

        # Wait for completion
        return_code = process.wait(timeout=timeout)

        # Wait for logging threads to finish
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        success = return_code == 0
        status = "success" if success else "failed"

        return {
            "success": success,
            "status": status,
            "return_code": return_code,
            "logs": logs,
            "command": cmd
        }

    except subprocess.TimeoutExpired:
        process.kill()
        return {
            "success": False,
            "status": "timeout",
            "return_code": -1,
            "logs": logs + [f"ERROR: Command timed out after {timeout} seconds"],
            "command": cmd
        }
    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "return_code": -1,
            "logs": logs + [f"ERROR: {str(e)}"],
            "command": cmd
        }