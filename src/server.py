"""MCP Server for BindCraft Scripts

Provides both synchronous and asynchronous (submit) APIs for protein binder design tools.
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import sys
import json

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
CLEAN_SCRIPTS_DIR = MCP_ROOT / "clean_scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CLEAN_SCRIPTS_DIR))

from jobs.manager import job_manager

# Create MCP server
mcp = FastMCP("bindcraft")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)

@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed
    """
    return job_manager.get_job_result(job_id)

@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)

@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)

@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)

# ==============================================================================
# Synchronous Tools (for fast operations < 10 min)
# ==============================================================================

@mcp.tool()
def quick_design(
    input_file: str,
    output_dir: Optional[str] = None,
    config: Optional[str] = None,
    num_designs: int = 1,
    chains: str = "A",
    binder_length: int = 130,
    device: int = 0,
    hotspot: Optional[str] = None
) -> dict:
    """
    Quick synchronous protein binder design from PDB structure.

    Use this for fast binder design (typically 1-10 minutes).
    For longer runs or multiple designs, use submit_async_design instead.

    Args:
        input_file: Path to input PDB file
        output_dir: Directory to save outputs (optional)
        config: Path to config file (optional)
        num_designs: Number of designs to generate (default: 1)
        chains: Chains to target (default: "A")
        binder_length: Length of binder sequence (default: 130)
        device: GPU device to use (default: 0)
        hotspot: Hotspot residues (optional)

    Returns:
        Dictionary with results including output files and design metrics
    """
    from use_case_1_quick_design import run_quick_design

    try:
        result = run_quick_design(
            input_file=input_file,
            output_file=output_dir,
            config=config,
            num_designs=num_designs,
            chains=chains,
            binder_length=binder_length,
            device=device,
            hotspot=hotspot
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        return {"status": "error", "error": f"Design failed: {str(e)}"}

@mcp.tool()
def monitor_progress(
    output_dir: str,
    detailed: bool = False,
    continuous: bool = False,
    interval: int = 30
) -> dict:
    """
    Monitor progress of running BindCraft jobs.

    Use this to check the status of jobs running in output directories.
    This tool reads log files and progress indicators without interfering with jobs.

    Args:
        output_dir: Directory containing the job to monitor
        detailed: Include detailed progress information (default: False)
        continuous: Enable continuous monitoring (default: False)
        interval: Monitoring interval in seconds (default: 30)

    Returns:
        Dictionary with job progress, status, and logs
    """
    from use_case_3_monitor_progress import run_monitor_progress

    try:
        result = run_monitor_progress(
            output_dir=output_dir,
            detailed=detailed,
            continuous=continuous,
            interval=interval
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"Directory not found: {e}"}
    except Exception as e:
        return {"status": "error", "error": f"Monitoring failed: {str(e)}"}

@mcp.tool()
def generate_config(
    input_file: str,
    output_file: Optional[str] = None,
    chains: str = "A",
    binder_length: int = 130,
    validate: bool = True,
    analysis_type: str = "basic"
) -> dict:
    """
    Generate BindCraft configuration files from PDB structures.

    This tool analyzes PDB files and creates configuration files for binder design.
    It's fast and doesn't require GPU resources.

    Args:
        input_file: Path to input PDB file
        output_file: Path for output config file (optional)
        chains: Chains to analyze (default: "A")
        binder_length: Target binder length (default: 130)
        validate: Validate generated config (default: True)
        analysis_type: Type of analysis (basic, detailed, advanced)

    Returns:
        Dictionary with generated configuration and validation results
    """
    from use_case_5_config_generator import run_config_generator

    try:
        result = run_config_generator(
            input_file=input_file,
            output_file=output_file,
            chains=chains,
            binder_length=binder_length,
            validate=validate,
            analysis_type=analysis_type
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        return {"status": "error", "error": f"Config generation failed: {str(e)}"}

# ==============================================================================
# Submit Tools (for long-running operations > 10 min)
# ==============================================================================

@mcp.tool()
def submit_async_design(
    input_file: str,
    output_dir: Optional[str] = None,
    config: Optional[str] = None,
    num_designs: int = 3,
    chains: str = "A",
    binder_length: int = 130,
    device: int = 0,
    hotspot: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit an asynchronous protein binder design job for background processing.

    This operation may take >10 minutes and runs in the background.
    Use this for multiple designs or when you want to submit and check back later.

    Args:
        input_file: Path to input PDB file
        output_dir: Directory to save outputs (optional)
        config: Path to config file (optional)
        num_designs: Number of designs to generate (default: 3)
        chains: Chains to target (default: "A")
        binder_length: Length of binder sequence (default: 130)
        device: GPU device to use (default: 0)
        hotspot: Hotspot residues (optional)
        job_name: Optional name for the job (for easier tracking)

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs
    """
    script_path = str(CLEAN_SCRIPTS_DIR / "use_case_2_async_submission.py")

    args = {
        "input": input_file,
        "num_designs": num_designs,
        "chains": chains,
        "binder_length": binder_length,
        "device": device
    }

    # Add optional arguments
    if output_dir:
        args["output"] = output_dir
    if config:
        args["config"] = config
    if hotspot:
        args["hotspot"] = hotspot

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"async_design_{Path(input_file).stem}"
    )

@mcp.tool()
def submit_batch_design(
    input_file: str,
    output_dir: Optional[str] = None,
    config: Optional[str] = None,
    num_designs: int = 1,
    max_concurrent: int = 3,
    chains: str = "A",
    binder_length: int = 130,
    device: int = 0,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch processing for multiple protein targets.

    Processes multiple targets from a batch file. This is ideal for:
    - Processing many structures at once
    - Large-scale binder design campaigns
    - Parallel processing of independent targets

    Args:
        input_file: Path to batch input file or directory with PDB files
        output_dir: Base directory for all outputs (optional)
        config: Path to config file (optional)
        num_designs: Number of designs per target (default: 1)
        max_concurrent: Maximum concurrent jobs (default: 3)
        chains: Chains to target (default: "A")
        binder_length: Length of binder sequence (default: 130)
        device: GPU device to use (default: 0)
        job_name: Optional name for the batch job

    Returns:
        Dictionary with job_id for tracking the batch job
    """
    script_path = str(CLEAN_SCRIPTS_DIR / "use_case_4_batch_design.py")

    args = {
        "input": input_file,
        "num_designs": num_designs,
        "max_concurrent": max_concurrent,
        "chains": chains,
        "binder_length": binder_length,
        "device": device
    }

    # Add optional arguments
    if output_dir:
        args["output"] = output_dir
    if config:
        args["config"] = config

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"batch_design_{Path(input_file).stem}"
    )

# ==============================================================================
# Utility Tools
# ==============================================================================

@mcp.tool()
def list_example_data() -> dict:
    """
    List available example data files for testing.

    Returns:
        Dictionary with available example files and their descriptions
    """
    try:
        examples_dir = MCP_ROOT / "examples" / "data"
        if not examples_dir.exists():
            return {
                "status": "error",
                "error": "Examples directory not found. Please ensure examples/data/ exists."
            }

        files = []
        for file_path in examples_dir.glob("*"):
            if file_path.is_file():
                files.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size_bytes": file_path.stat().st_size,
                    "type": file_path.suffix
                })

        return {
            "status": "success",
            "example_files": files,
            "examples_directory": str(examples_dir),
            "total_files": len(files)
        }
    except Exception as e:
        return {"status": "error", "error": f"Failed to list examples: {str(e)}"}

@mcp.tool()
def get_default_configs() -> dict:
    """
    Get information about available configuration files.

    Returns:
        Dictionary with available config files and their settings
    """
    try:
        configs_dir = MCP_ROOT / "configs"
        if not configs_dir.exists():
            return {
                "status": "error",
                "error": "Configs directory not found. Please ensure configs/ exists."
            }

        configs = []
        for config_file in configs_dir.glob("*.json"):
            try:
                with open(config_file) as f:
                    config_data = json.load(f)
                configs.append({
                    "name": config_file.name,
                    "path": str(config_file),
                    "description": config_data.get("description", "No description"),
                    "settings": config_data
                })
            except Exception as e:
                configs.append({
                    "name": config_file.name,
                    "path": str(config_file),
                    "description": f"Error loading config: {e}",
                    "settings": None
                })

        return {
            "status": "success",
            "config_files": configs,
            "configs_directory": str(configs_dir),
            "total_configs": len(configs)
        }
    except Exception as e:
        return {"status": "error", "error": f"Failed to list configs: {str(e)}"}

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()