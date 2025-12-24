#!/usr/bin/env python3
"""
Script: use_case_3_monitor_progress.py
Description: Monitor progress of running BindCraft jobs

Original Use Case: examples/use_case_3_monitor_progress.py
Dependencies Removed: Inlined MCP decorators and framework dependencies

Usage:
    python clean_scripts/use_case_3_monitor_progress.py --output <output_dir>

Example:
    python clean_scripts/use_case_3_monitor_progress.py --output results/async_job --detailed
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import time
import sys
from pathlib import Path
from typing import Union, Optional, Dict, Any

# Add lib to path for shared utilities
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from lib.io import load_json

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_MONITOR_CONFIG = {
    "refresh_interval": 30,  # seconds
    "max_iterations": 120,   # 1 hour with 30s intervals
    "detailed": False
}

# ==============================================================================
# Core Functions (main logic extracted from use case)
# ==============================================================================
def analyze_log_file(log_file: Path) -> Dict[str, Any]:
    """Analyze BindCraft log file for progress information."""
    status_info = {
        "job_status": "unknown",
        "current_phase": "unknown",
        "progress_info": "",
        "error_detected": False,
        "completion_detected": False
    }

    if not log_file.exists():
        status_info["job_status"] = "not_started"
        status_info["progress_info"] = "Log file not found"
        return status_info

    try:
        # Read last 100 lines for recent progress
        lines = log_file.read_text().strip().split('\n')
        recent_lines = lines[-100:] if len(lines) > 100 else lines

        # Analyze log content
        for line in recent_lines:
            line = line.lower()

            # Check for completion
            if "design completed" in line or "all designs finished" in line:
                status_info["completion_detected"] = True
                status_info["job_status"] = "completed"
                status_info["current_phase"] = "completed"

            # Check for errors
            elif "error" in line or "failed" in line or "exception" in line:
                status_info["error_detected"] = True
                status_info["job_status"] = "error"
                status_info["progress_info"] = line.strip()

            # Check for current phase
            elif "af2 hallucination" in line or "alphafold" in line:
                status_info["current_phase"] = "af2_hallucination"
                status_info["job_status"] = "running"
            elif "mpnn" in line or "proteinmpnn" in line:
                status_info["current_phase"] = "mpnn_design"
                status_info["job_status"] = "running"
            elif "pyrosetta" in line or "scoring" in line:
                status_info["current_phase"] = "scoring"
                status_info["job_status"] = "running"

        # If no specific status found but file exists, assume running
        if status_info["job_status"] == "unknown" and len(lines) > 0:
            status_info["job_status"] = "running"

    except Exception as e:
        status_info["error_detected"] = True
        status_info["progress_info"] = f"Error reading log file: {str(e)}"

    return status_info


def count_output_files(output_dir: Path) -> Dict[str, Any]:
    """Count design output files to track progress."""
    stats = {
        "accepted_designs": 0,
        "rejected_designs": 0,
        "mpnn_files": 0,
        "trajectory_files": 0,
        "accepted_files": [],
        "directories_found": []
    }

    # Count accepted designs
    accepted_dir = output_dir / "Accepted"
    if accepted_dir.exists():
        stats["directories_found"].append("Accepted")
        accepted_files = list(accepted_dir.glob("*.pdb"))
        stats["accepted_designs"] = len(accepted_files)
        stats["accepted_files"] = [f.name for f in accepted_files]

    # Count rejected designs
    rejected_dir = output_dir / "Rejected"
    if rejected_dir.exists():
        stats["directories_found"].append("Rejected")
        rejected_files = list(rejected_dir.glob("*.pdb"))
        stats["rejected_designs"] = len(rejected_files)

    # Count MPNN files
    mpnn_dir = output_dir / "MPNN"
    if mpnn_dir.exists():
        stats["directories_found"].append("MPNN")
        mpnn_files = list(mpnn_dir.glob("*"))
        stats["mpnn_files"] = len(mpnn_files)

    # Count trajectory files
    trajectory_dir = output_dir / "Trajectory"
    if trajectory_dir.exists():
        stats["directories_found"].append("Trajectory")
        trajectory_files = list(trajectory_dir.glob("*"))
        stats["trajectory_files"] = len(trajectory_files)

    return stats


def get_target_configuration(output_dir: Path) -> Dict[str, Any]:
    """Load target configuration if available."""
    config_info = {
        "target_found": False,
        "target_name": "Unknown",
        "num_designs": "Unknown",
        "target_chains": "Unknown"
    }

    settings_file = output_dir / "target_settings.json"
    if settings_file.exists():
        try:
            settings = load_json(settings_file)
            config_info.update({
                "target_found": True,
                "target_name": settings.get("binder_name", "Unknown"),
                "num_designs": settings.get("num_designs", "Unknown"),
                "target_chains": settings.get("target_chains", "Unknown"),
                "target_pdb": settings.get("target_pdb", "Unknown")
            })
        except Exception:
            pass

    return config_info


def run_monitor_progress(
    output_dir: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for monitoring BindCraft job progress.

    Args:
        output_dir: Path to job output directory
        config: Configuration dict (uses DEFAULT_MONITOR_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - result: Current job status and progress information
            - monitoring_info: Details about monitoring session

    Example:
        >>> result = run_monitor_progress("results/async_job")
        >>> print(f"Job status: {result['result']['job_status']}")
    """
    # Setup
    output_dir = Path(output_dir)
    final_config = {**DEFAULT_MONITOR_CONFIG, **(config or {}), **kwargs}

    if not output_dir.exists():
        return {
            "success": False,
            "error": f"Output directory not found: {output_dir}",
            "result": {"job_status": "not_found"}
        }

    try:
        # Check for log file
        log_file = output_dir / "bindcraft_run.log"

        # Analyze current status
        log_analysis = analyze_log_file(log_file)
        file_stats = count_output_files(output_dir)
        config_info = get_target_configuration(output_dir)

        # Combine information
        result = {
            "job_status": log_analysis["job_status"],
            "current_phase": log_analysis["current_phase"],
            "progress_info": log_analysis["progress_info"],
            "error_detected": log_analysis["error_detected"],
            "completion_detected": log_analysis["completion_detected"],
            "output_statistics": file_stats,
            "target_configuration": config_info,
            "log_file_exists": log_file.exists(),
            "log_file_path": str(log_file)
        }

        return {
            "success": True,
            "result": result,
            "monitoring_info": {
                "output_directory": str(output_dir),
                "monitoring_time": "Current snapshot",
                "config": final_config
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Monitoring failed: {str(e)}",
            "result": {"job_status": "error"}
        }


def run_continuous_monitoring(
    output_dir: Union[str, Path],
    refresh_interval: int = 30,
    max_iterations: int = 120
) -> None:
    """Run continuous monitoring with periodic updates."""
    output_dir = Path(output_dir)

    print(f"🔍 Starting continuous monitoring of: {output_dir}")
    print(f"   Refresh interval: {refresh_interval} seconds")
    print(f"   Max duration: {max_iterations * refresh_interval} seconds")
    print("   Press Ctrl+C to stop\n")

    iteration = 0
    try:
        while iteration < max_iterations:
            result = run_monitor_progress(output_dir)

            if result["success"]:
                status_info = result["result"]

                print(f"--- Update {iteration + 1} ---")
                print(f"Status: {status_info['job_status']}")
                print(f"Phase: {status_info['current_phase']}")

                if status_info['output_statistics']['accepted_designs'] > 0:
                    stats = status_info['output_statistics']
                    print(f"Progress: {stats['accepted_designs']} accepted, {stats['rejected_designs']} rejected")

                if status_info['error_detected']:
                    print(f"❌ Error detected: {status_info['progress_info']}")
                    break

                if status_info['completion_detected']:
                    print("✅ Job completed!")
                    break

                if status_info['progress_info']:
                    print(f"Info: {status_info['progress_info']}")

                print()

            iteration += 1
            if iteration < max_iterations:
                time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")


# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--output', '-o', required=True, help='Job output directory to monitor')
    parser.add_argument('--detailed', action='store_true', help='Show detailed information')
    parser.add_argument('--continuous', action='store_true', help='Continuous monitoring mode')
    parser.add_argument('--interval', type=int, default=30, help='Refresh interval for continuous mode (seconds)')
    parser.add_argument('--max-time', type=int, default=3600, help='Maximum monitoring time (seconds)')

    args = parser.parse_args()

    if args.continuous:
        max_iterations = args.max_time // args.interval
        run_continuous_monitoring(args.output, args.interval, max_iterations)
    else:
        # Single status check
        result = run_monitor_progress(
            output_dir=args.output,
            config={"detailed": args.detailed}
        )

        if result["success"]:
            status_info = result["result"]
            config_info = status_info["target_configuration"]
            stats = status_info["output_statistics"]

            print(f"📊 Job Status Report")
            print(f"   Directory: {args.output}")
            print(f"   Status: {status_info['job_status']}")
            print(f"   Phase: {status_info['current_phase']}")

            if config_info["target_found"]:
                print(f"   Target: {config_info['target_name']}")
                print(f"   Expected designs: {config_info['num_designs']}")
                print(f"   Target chains: {config_info['target_chains']}")

            print(f"   Progress: {stats['accepted_designs']} accepted, {stats['rejected_designs']} rejected")

            if args.detailed:
                print(f"   MPNN files: {stats['mpnn_files']}")
                print(f"   Trajectory files: {stats['trajectory_files']}")
                print(f"   Directories found: {', '.join(stats['directories_found'])}")
                print(f"   Log file exists: {status_info['log_file_exists']}")

                if stats['accepted_files']:
                    print(f"   Accepted files: {', '.join(stats['accepted_files'])}")

            if status_info['error_detected']:
                print(f"   ❌ Error: {status_info['progress_info']}")
            elif status_info['completion_detected']:
                print(f"   ✅ Completed!")
            elif status_info['job_status'] == 'running':
                print(f"   🔄 Running...")

            if status_info['progress_info'] and not status_info['error_detected']:
                print(f"   Info: {status_info['progress_info']}")

        else:
            print(f"❌ Monitoring failed: {result['error']}")
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())