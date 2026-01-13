"""
BindCraft binder design tools.

This MCP Server provides tools for designing de novo protein binders:
1. bindcraft_design_binder: Run full BindCraft binder design pipeline (synchronous)
2. bindcraft_submit: Submit a BindCraft job asynchronously (returns immediately)
3. bindcraft_check_status: Check status and results of a submitted job

The tools use:
- AlphaFold2 for binder hallucination and structure prediction
- ProteinMPNN for sequence optimization
- PyRosetta for structure relaxation and interface scoring
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Annotated, Literal, Optional

from fastmcp import FastMCP
from loguru import logger

# MCP server instance
bindcraft_design_mcp = FastMCP(name="bindcraft_design")


def _get_bindcraft_path() -> Path:
    """Get the BindCraft scripts path."""
    # src/tools/bindcraft_design.py -> src -> bindcraft_mcp -> scripts
    src_dir = Path(__file__).parent.parent.absolute()
    bindcraft_path = src_dir.parent / "scripts"

    if not bindcraft_path.exists():
        logger.error(f"BindCraft scripts not found at {bindcraft_path}")
        raise FileNotFoundError(
            f"BindCraft scripts not found at {bindcraft_path}. "
            "Please ensure the scripts directory exists."
        )
    return bindcraft_path


def _get_default_settings_paths(bindcraft_path: Path) -> dict:
    """Get default paths for filter and advanced settings."""
    repo_path = bindcraft_path.parent / "repo" / "BindCraft"
    return {
        "filters": str(repo_path / "settings_filters" / "default_filters.json"),
        "advanced": str(repo_path / "settings_advanced" / "default_4stage_multimer.json"),
    }


def _resolve_path(path: Optional[str]) -> Optional[str]:
    """Resolve a path to absolute."""
    if path is None:
        return None
    return str(Path(path).resolve())


# Required fields for run_bindcraft.py target settings
REQUIRED_SETTINGS_FIELDS = [
    "design_path",
    "binder_name",
    "starting_pdb",
    "chains",
    "lengths",
    "number_of_final_designs"
]


def _validate_target_settings(settings: dict) -> list[str]:
    """Validate target settings against run_bindcraft.py requirements.

    Returns a list of error messages. Empty list means validation passed.
    """
    errors = []

    # Check required fields
    for field in REQUIRED_SETTINGS_FIELDS:
        if field not in settings:
            errors.append(f"Missing required field: '{field}'")

    # Validate 'lengths' format (must be array with 2 elements)
    if "lengths" in settings:
        lengths = settings["lengths"]
        if not isinstance(lengths, list):
            errors.append(
                f"'lengths' must be an array [min, max], not {type(lengths).__name__}. "
                f"Got: {lengths}"
            )
        elif len(lengths) != 2:
            errors.append(
                f"'lengths' must have exactly 2 elements [min_length, max_length]. "
                f"Got {len(lengths)} elements: {lengths}"
            )

    # Validate 'starting_pdb' exists
    if "starting_pdb" in settings:
        pdb_path = Path(settings["starting_pdb"])
        if not pdb_path.exists():
            errors.append(f"Target PDB file not found: {settings['starting_pdb']}")

    # Validate 'number_of_final_designs' is positive
    if "number_of_final_designs" in settings:
        num_designs = settings["number_of_final_designs"]
        if not isinstance(num_designs, int) or num_designs <= 0:
            errors.append(
                f"'number_of_final_designs' must be a positive integer. Got: {num_designs}"
            )

    return errors


def _log_stream(stream, logs: list[str], prefix: str = ""):
    """Collect output from a stream and print in real-time."""
    for line in iter(stream.readline, ""):
        line = line.rstrip()
        if line:
            logs.append(line)
            logger.info(f"{prefix}{line}")


def _run_command(
    cmd: list[str],
    device: int = 0,
    cwd: Optional[str] = None,
) -> dict:
    """Run a command with proper environment setup."""
    run_env = os.environ.copy()
    run_env["CUDA_VISIBLE_DEVICES"] = str(device)
    run_env["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
    run_env["PYTHONUNBUFFERED"] = "1"

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    logger.debug(f"Executing command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=run_env,
        cwd=cwd,
        bufsize=1,
    )

    stdout_thread = threading.Thread(
        target=_log_stream,
        args=(process.stdout, stdout_lines, "[BindCraft] "),
    )
    stderr_thread = threading.Thread(
        target=_log_stream,
        args=(process.stderr, stderr_lines, "[BindCraft stderr] "),
    )

    stdout_thread.start()
    stderr_thread.start()

    process.wait()
    stdout_thread.join()
    stderr_thread.join()

    if process.stdout:
        process.stdout.close()
    if process.stderr:
        process.stderr.close()

    logger.debug(f"Command completed with return code: {process.returncode}")

    return {
        "success": process.returncode == 0,
        "return_code": process.returncode,
        "stdout": "\n".join(stdout_lines),
        "stderr": "\n".join(stderr_lines),
    }


@bindcraft_design_mcp.tool
def bindcraft_design_binder(
    settings_json: Annotated[str, "Path to target settings JSON file"],
    filters_json: Annotated[Optional[str], "Path to filters JSON file. If None, uses default"] = None,
    advanced_json: Annotated[Optional[str], "Path to advanced settings JSON file. If None, uses default"] = None,
    device: Annotated[int, "GPU device number"] = 0,
    log_level: Annotated[
        Literal["DEBUG", "INFO", "WARNING", "ERROR"],
        "Logging level for the design process"
    ] = "INFO",
) -> dict:
    """
    Run BindCraft binder design pipeline.

    This tool designs de novo protein binders against a target protein using:
    1. **AF2 Hallucination**: Generate binder backbone conformations guided by target structure
    2. **MPNN Sequence Design**: Optimize amino acid sequences for designed backbones
    3. **AF2 Prediction**: Validate binder-target complexes with structure prediction
    4. **PyRosetta Analysis**: Score interface quality, energy, and structural metrics

    Target Settings JSON Format:
    ```json
    {
        "design_path": "output/directory/",
        "binder_name": "MyBinder",
        "starting_pdb": "path/to/target.pdb",
        "chains": "A",
        "target_hotspot_residues": "56,78,90",
        "lengths": [65, 150],
        "number_of_final_designs": 100
    }
    ```

    Fields:
    - design_path: Output directory for designed binders
    - binder_name: Prefix name for designed binders
    - starting_pdb: Target protein PDB structure
    - chains: Target chain(s) to design binder against
    - target_hotspot_residues: Residue numbers to target (comma-separated)
    - lengths: [min_length, max_length] range for binder length
    - number_of_final_designs: Target number of accepted designs

    Input: Path to target settings JSON file
    Output: Dictionary with design status, output paths, and statistics
    """
    logger.info(f"bindcraft_design_binder called with settings={settings_json}")

    try:
        bindcraft_path = _get_bindcraft_path()
        defaults = _get_default_settings_paths(bindcraft_path)

        # Resolve paths
        settings_json = _resolve_path(settings_json)
        filters_json = _resolve_path(filters_json) or defaults["filters"]
        advanced_json = _resolve_path(advanced_json) or defaults["advanced"]

        # Validate settings file exists
        if not Path(settings_json).exists():
            logger.error(f"Settings file not found: {settings_json}")
            raise FileNotFoundError(f"Settings file not found: {settings_json}")

        # Load settings to get output path
        with open(settings_json, 'r') as f:
            target_settings = json.load(f)

        # Validate settings before running
        validation_errors = _validate_target_settings(target_settings)
        if validation_errors:
            error_msg = "Settings validation failed:\n  - " + "\n  - ".join(validation_errors)
            logger.error(error_msg)
            return {
                "status": "error",
                "error_message": error_msg,
                "validation_errors": validation_errors,
                "settings_json": settings_json,
                "hint": "Required fields: design_path, binder_name, starting_pdb, chains, lengths (array), number_of_final_designs"
            }

        output_dir = Path(target_settings.get("design_path", "output")).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Target: {target_settings.get('binder_name')}")
        logger.info(f"Output directory: {output_dir}")

        # Build command
        cmd = [
            sys.executable,
            str(bindcraft_path / "run_bindcraft.py"),
            f"--settings={settings_json}",
            f"--filters={filters_json}",
            f"--advanced={advanced_json}",
            f"--log-level={log_level}",
        ]

        logger.info(f"Running BindCraft on GPU {device}")

        # Run design
        result = _run_command(cmd, device=device, cwd=str(bindcraft_path))

        # Collect output statistics
        output_stats = {
            "accepted_designs": 0,
            "rejected_designs": 0,
            "trajectories": 0,
        }

        if result["success"]:
            # Count output files
            accepted_dir = output_dir / "Accepted"
            rejected_dir = output_dir / "Rejected"
            trajectory_dir = output_dir / "Trajectory" / "Relaxed"

            if accepted_dir.exists():
                output_stats["accepted_designs"] = len(list(accepted_dir.glob("*.pdb")))
            if rejected_dir.exists():
                output_stats["rejected_designs"] = len(list(rejected_dir.glob("*.pdb")))
            if trajectory_dir.exists():
                output_stats["trajectories"] = len(list(trajectory_dir.glob("*.pdb")))

            logger.info(f"Design completed. Accepted: {output_stats['accepted_designs']}, "
                       f"Rejected: {output_stats['rejected_designs']}, "
                       f"Trajectories: {output_stats['trajectories']}")
        else:
            logger.error(f"Design failed with return code {result['return_code']}")

        return {
            "status": "success" if result["success"] else "error",
            "settings_json": settings_json,
            "filters_json": filters_json,
            "advanced_json": advanced_json,
            "output_dir": str(output_dir),
            "target_name": target_settings.get("binder_name"),
            "target_pdb": target_settings.get("starting_pdb"),
            "target_chains": target_settings.get("chains"),
            "binder_length_range": target_settings.get("lengths"),
            "target_designs": target_settings.get("number_of_final_designs"),
            "statistics": output_stats,
            "return_code": result["return_code"],
            "device": device,
            "stdout_preview": result["stdout"][-3000:] if len(result["stdout"]) > 3000 else result["stdout"],
            "stderr_preview": result["stderr"][-2000:] if len(result["stderr"]) > 2000 else result["stderr"],
        }

    except Exception as e:
        logger.exception(f"Exception during binder design: {e}")
        return {
            "status": "error",
            "error_message": str(e),
            "settings_json": settings_json,
        }


@bindcraft_design_mcp.tool
def bindcraft_submit(
    settings_json: Annotated[Optional[str], "Path to target settings JSON file"] = None,
    target_pdb: Annotated[Optional[str], "Path to target protein PDB file (for simple submission)"] = None,
    output_dir: Annotated[Optional[str], "Output directory for designed binders (required if using target_pdb)"] = None,
    binder_name: Annotated[Optional[str], "Name prefix for designed binders (required if using target_pdb)"] = None,
    target_chains: Annotated[str, "Target chain(s) to design binder against (e.g., 'A' or 'A,B')"] = "A",
    hotspot_residues: Annotated[Optional[str], "Target residue numbers, comma-separated (e.g., '56,78,90'). If None, auto-detect"] = None,
    min_binder_length: Annotated[int, "Minimum binder length in residues"] = 65,
    max_binder_length: Annotated[int, "Maximum binder length in residues"] = 150,
    num_designs: Annotated[int, "Target number of final accepted designs"] = 10,
    filters_json: Annotated[Optional[str], "Path to filters JSON file. If None, uses default"] = None,
    advanced_json: Annotated[Optional[str], "Path to advanced settings JSON file. If None, uses default"] = None,
    device: Annotated[int, "GPU device number"] = 0,
    log_level: Annotated[
        Literal["DEBUG", "INFO", "WARNING", "ERROR"],
        "Logging level for the design process"
    ] = "INFO",
) -> dict:
    """
    Submit a BindCraft binder design job asynchronously.

    This tool submits a BindCraft job and returns immediately with a 'submitted' status.
    The job runs in the background and results can be queried using bindcraft_check_status.

    Two submission modes:

    1. **Config File Mode**: Provide a settings_json file
       - settings_json: Path to complete target settings JSON

    2. **Simple PDB Mode**: Provide target_pdb and basic parameters
       - target_pdb: Path to target protein PDB file
       - output_dir: Output directory for results
       - binder_name: Name prefix for designs
       - Optional: chains, hotspot_residues, lengths, num_designs, etc.

    Target Settings JSON Format (for Config File Mode):
    ```json
    {
        "design_path": "output/directory/",
        "binder_name": "MyBinder",
        "starting_pdb": "path/to/target.pdb",
        "chains": "A",
        "target_hotspot_residues": "56,78,90",
        "lengths": [65, 150],
        "number_of_final_designs": 100
    }
    ```

    The tool will:
    1. Validate inputs and create settings if needed
    2. Launch the BindCraft process in the background
    3. Return immediately with submission status and output directory
    4. Job continues running independently

    Use bindcraft_check_status with the returned output_dir to monitor progress.

    Input: Either settings_json OR (target_pdb + output_dir + binder_name)
    Output: Dictionary with status='submitted', output_dir, and job info
    """
    logger.info("bindcraft_submit called")

    try:
        bindcraft_path = _get_bindcraft_path()
        defaults = _get_default_settings_paths(bindcraft_path)

        # Determine submission mode
        if settings_json is not None:
            # Config File Mode
            logger.info(f"Config file mode: {settings_json}")
            settings_json = _resolve_path(settings_json)

            if not Path(settings_json).exists():
                logger.error(f"Settings file not found: {settings_json}")
                raise FileNotFoundError(f"Settings file not found: {settings_json}")

            # Load settings to get output path
            with open(settings_json, 'r') as f:
                target_settings = json.load(f)

            output_path = Path(target_settings.get("design_path", "output")).resolve()
            output_path.mkdir(parents=True, exist_ok=True)

            settings_json_path = Path(settings_json)

        elif target_pdb is not None and output_dir is not None and binder_name is not None:
            # Simple PDB Mode
            logger.info(f"Simple PDB mode: {target_pdb}")
            target_pdb = _resolve_path(target_pdb)
            output_path = Path(output_dir).resolve()

            # Validate target PDB exists
            if not Path(target_pdb).exists():
                logger.error(f"Target PDB not found: {target_pdb}")
                raise FileNotFoundError(f"Target PDB not found: {target_pdb}")

            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)

            # Create target settings JSON
            target_settings = {
                "design_path": str(output_path),
                "binder_name": binder_name,
                "starting_pdb": target_pdb,
                "chains": target_chains,
                "target_hotspot_residues": hotspot_residues or "",
                "lengths": [min_binder_length, max_binder_length],
                "number_of_final_designs": num_designs,
            }

            settings_json_path = output_path / "target_settings.json"
            with open(settings_json_path, 'w') as f:
                json.dump(target_settings, f, indent=2)

            logger.info(f"Created target settings at: {settings_json_path}")

        else:
            error_msg = "Must provide either 'settings_json' OR ('target_pdb' + 'output_dir' + 'binder_name')"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate settings before submission
        validation_errors = _validate_target_settings(target_settings)
        if validation_errors:
            error_msg = "Settings validation failed:\n  - " + "\n  - ".join(validation_errors)
            logger.error(error_msg)
            return {
                "status": "error",
                "error_message": error_msg,
                "validation_errors": validation_errors,
                "hint": "Required fields: design_path, binder_name, starting_pdb, chains, lengths (array), number_of_final_designs"
            }

        # Resolve filter and advanced settings paths
        filters_json = _resolve_path(filters_json) or defaults["filters"]
        advanced_json = _resolve_path(advanced_json) or defaults["advanced"]

        # Build command
        cmd = [
            sys.executable,
            str(bindcraft_path / "run_bindcraft.py"),
            f"--settings={settings_json_path}",
            f"--filters={filters_json}",
            f"--advanced={advanced_json}",
            f"--log-level={log_level}",
        ]

        # Setup environment
        run_env = os.environ.copy()
        run_env["CUDA_VISIBLE_DEVICES"] = str(device)
        run_env["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
        run_env["PYTHONUNBUFFERED"] = "1"

        # Create log file for the background process
        log_file = output_path / "bindcraft_run.log"

        logger.info(f"Submitting BindCraft job on GPU {device}")
        logger.debug(f"Command: {' '.join(cmd)}")
        logger.info(f"Output directory: {output_path}")
        logger.info(f"Log file: {log_file}")

        # Launch process in background
        with open(log_file, 'w') as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=run_env,
                cwd=str(bindcraft_path),
                start_new_session=True,  # Detach from parent
            )

        logger.info(f"Job submitted successfully. PID: {process.pid}")

        return {
            "status": "submitted",
            "message": "BindCraft job submitted successfully. Use bindcraft_check_status to monitor progress.",
            "output_dir": str(output_path),
            "settings_json": str(settings_json_path),
            "filters_json": filters_json,
            "advanced_json": advanced_json,
            "log_file": str(log_file),
            "pid": process.pid,
            "device": device,
            "target_settings": target_settings,
        }

    except Exception as e:
        logger.exception(f"Exception during job submission: {e}")
        return {
            "status": "error",
            "error_message": str(e),
        }


@bindcraft_design_mcp.tool
def bindcraft_check_status(
    output_dir: Annotated[str, "Path to BindCraft output directory"],
) -> dict:
    """
    Check the status of a BindCraft design run.

    This tool inspects an existing output directory to report:
    - Job status: running, completed, failed, or unknown
    - Number of completed trajectories
    - Number of accepted designs
    - Number of rejected designs
    - CSV statistics files available
    - Result summary (if job is finished)

    The tool determines job status by:
    1. Checking if the log file exists and parsing it for completion markers
    2. Detecting success/failure indicators in the log
    3. Providing detailed summary when job is complete

    Use this to monitor long-running design jobs or inspect completed runs.

    Input: Path to BindCraft output directory
    Output: Dictionary with job status, design statistics, and summary if finished
    """
    logger.info(f"bindcraft_check_status called for: {output_dir}")

    try:
        output_dir = Path(output_dir).resolve()

        if not output_dir.exists():
            return {
                "status": "error",
                "error_message": f"Output directory not found: {output_dir}",
            }

        # Check for log file and determine job status
        log_file = output_dir / "bindcraft_run.log"
        job_status = "unknown"
        log_tail = []
        error_messages = []

        if log_file.exists():
            import time

            # Check file modification time to detect if still running
            log_mtime = log_file.stat().st_mtime
            current_time = time.time()
            time_since_update = current_time - log_mtime

            # Read log file to check for completion markers
            try:
                with open(log_file, 'r') as f:
                    log_lines = f.readlines()

                # Get last 50 lines for inspection
                log_tail = [line.strip() for line in log_lines[-50:] if line.strip()]

                # Look for completion/error markers in the log
                log_content_lower = ''.join(log_lines).lower()

                # Check for completion indicators
                has_completion = any(marker in log_content_lower for marker in [
                    'design completed',
                    'bindcraft finished',
                    'all designs completed',
                    'pipeline completed successfully'
                ])

                # Check for error indicators
                has_error = any(marker in log_content_lower for marker in [
                    'error:',
                    'exception:',
                    'traceback',
                    'failed:',
                    'fatal'
                ])

                # Extract error messages if present
                if has_error:
                    for line in log_lines[-100:]:
                        line_lower = line.lower()
                        if any(err in line_lower for err in ['error:', 'exception:', 'failed:', 'fatal:']):
                            error_messages.append(line.strip())

                # Determine status
                if has_completion and not has_error:
                    job_status = "completed"
                elif has_error:
                    job_status = "failed"
                elif time_since_update < 300:  # Updated within last 5 minutes
                    job_status = "running"
                elif time_since_update < 3600:  # Updated within last hour
                    job_status = "possibly_running"
                else:
                    job_status = "stalled_or_completed"

            except Exception as e:
                logger.warning(f"Could not parse log file: {e}")
                job_status = "unknown"
        else:
            job_status = "not_started"

        # Count files in various directories
        stats = {
            "accepted_designs": 0,
            "accepted_ranked": 0,
            "rejected_designs": 0,
            "trajectories_relaxed": 0,
            "trajectories_clashing": 0,
            "trajectories_low_confidence": 0,
            "mpnn_designs": 0,
        }

        # Check directories
        dirs_to_check = {
            "accepted_designs": output_dir / "Accepted",
            "accepted_ranked": output_dir / "Accepted" / "Ranked",
            "rejected_designs": output_dir / "Rejected",
            "trajectories_relaxed": output_dir / "Trajectory" / "Relaxed",
            "trajectories_clashing": output_dir / "Trajectory" / "Clashing",
            "trajectories_low_confidence": output_dir / "Trajectory" / "LowConfidence",
            "mpnn_designs": output_dir / "MPNN" / "Relaxed",
        }

        for key, dir_path in dirs_to_check.items():
            if dir_path.exists():
                stats[key] = len(list(dir_path.glob("*.pdb")))

        # Check for CSV files
        csv_files = {}
        for csv_name in ["trajectory_stats.csv", "mpnn_design_stats.csv", "final_design_stats.csv", "failure_csv.csv"]:
            csv_path = output_dir / csv_name
            if csv_path.exists():
                csv_files[csv_name] = str(csv_path)

        # List top accepted designs
        accepted_files = []
        accepted_dir = output_dir / "Accepted"
        if accepted_dir.exists():
            accepted_files = sorted([f.name for f in accepted_dir.glob("*.pdb")])[:20]

        # Check for target settings
        target_settings = None
        settings_path = output_dir / "target_settings.json"
        if settings_path.exists():
            with open(settings_path) as f:
                target_settings = json.load(f)

        # Build response
        response = {
            "status": "success",
            "job_status": job_status,
            "output_dir": str(output_dir),
            "statistics": stats,
            "csv_files": csv_files,
            "accepted_designs": accepted_files,
            "target_settings": target_settings,
            "total_trajectories": stats["trajectories_relaxed"] + stats["trajectories_clashing"] + stats["trajectories_low_confidence"],
            "log_file": str(log_file) if log_file.exists() else None,
        }

        # Add summary if job is finished (completed or failed)
        if job_status in ["completed", "failed"]:
            summary = _generate_job_summary(
                job_status=job_status,
                stats=stats,
                target_settings=target_settings,
                log_tail=log_tail[-20:],
                error_messages=error_messages[-10:],
            )
            response["summary"] = summary

        return response

    except Exception as e:
        logger.exception(f"Exception checking status: {e}")
        return {
            "status": "error",
            "error_message": str(e),
            "output_dir": str(output_dir),
        }


def _generate_job_summary(
    job_status: str,
    stats: dict,
    target_settings: Optional[dict],
    log_tail: list[str],
    error_messages: list[str],
) -> dict:
    """Generate a comprehensive summary for completed/failed jobs."""
    summary = {
        "job_status": job_status,
        "completion_status": "Success" if job_status == "completed" else "Failed",
    }

    # Add target information
    if target_settings:
        summary["target"] = {
            "binder_name": target_settings.get("binder_name"),
            "target_pdb": target_settings.get("starting_pdb"),
            "chains": target_settings.get("chains"),
            "hotspot_residues": target_settings.get("target_hotspot_residues"),
            "binder_length_range": target_settings.get("lengths"),
            "requested_designs": target_settings.get("number_of_final_designs"),
        }

    # Add results summary
    total_trajectories = stats["trajectories_relaxed"] + stats["trajectories_clashing"] + stats["trajectories_low_confidence"]
    total_accepted = stats["accepted_designs"]
    total_rejected = stats["rejected_designs"]

    summary["results"] = {
        "accepted_designs": total_accepted,
        "rejected_designs": total_rejected,
        "total_evaluated": total_accepted + total_rejected,
        "trajectories_generated": total_trajectories,
        "trajectories_relaxed": stats["trajectories_relaxed"],
        "trajectories_clashing": stats["trajectories_clashing"],
        "trajectories_low_confidence": stats["trajectories_low_confidence"],
        "mpnn_designs": stats["mpnn_designs"],
    }

    # Add success metrics
    if target_settings and total_accepted > 0:
        requested = target_settings.get("number_of_final_designs", 0)
        if requested > 0:
            success_rate = (total_accepted / requested) * 100
            summary["success_rate"] = f"{success_rate:.1f}% of target ({total_accepted}/{requested})"

        if total_accepted + total_rejected > 0:
            acceptance_rate = (total_accepted / (total_accepted + total_rejected)) * 100
            summary["acceptance_rate"] = f"{acceptance_rate:.1f}% ({total_accepted}/{total_accepted + total_rejected})"

    # Add message based on status
    if job_status == "completed":
        if total_accepted > 0:
            summary["message"] = f"Design completed successfully with {total_accepted} accepted binder(s)."
        else:
            summary["message"] = f"Design completed but no binders passed acceptance criteria. Generated {total_trajectories} trajectories, all were rejected."
    elif job_status == "failed":
        summary["message"] = "Design job failed. Check error messages and log file for details."
        if error_messages:
            summary["recent_errors"] = error_messages

    # Add log tail
    if log_tail:
        summary["log_tail"] = log_tail

    return summary
