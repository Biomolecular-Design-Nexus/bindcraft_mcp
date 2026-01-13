"""
BindCraft configuration tools.

This MCP Server provides tools for generating and validating BindCraft configurations:
1. generate_config: Generate BindCraft configuration files from PDB structures
2. validate_config: Validate configuration files before job submission

These tools help prepare configuration files for binder design jobs without
requiring GPU resources, making them fast and suitable for pre-flight checks.
"""

import sys
from pathlib import Path
from typing import Annotated, Optional

from fastmcp import FastMCP
from loguru import logger

# Setup paths
SCRIPT_DIR = Path(__file__).parent.parent.resolve()  # src/
MCP_ROOT = SCRIPT_DIR.parent  # bindcraft_mcp/
CLEAN_SCRIPTS_DIR = MCP_ROOT / "clean_scripts"
sys.path.insert(0, str(CLEAN_SCRIPTS_DIR))

# MCP server instance
bindcraft_config_mcp = FastMCP(name="bindcraft_config")


@bindcraft_config_mcp.tool
def generate_config(
    input_file: Annotated[str, "Path to input PDB file"],
    output_file: Annotated[Optional[str], "Path for output directory"] = None,
    chains: Annotated[str, "Chains to target"] = "A",
    binder_length: Annotated[int, "Target binder length"] = 130,
    validate: Annotated[bool, "Validate generated config"] = True,
    analysis_type: Annotated[str, "Type of analysis (basic, detailed, advanced)"] = "basic",
    num_designs: Annotated[int, "Number of designs to generate"] = 1,
    binder_name: Annotated[str, "Name prefix for designed binders"] = "Binder",
    hotspot: Annotated[Optional[str], "Hotspot residues to target, comma-separated"] = None
) -> dict:
    """
    Generate BindCraft configuration files from PDB structures.

    This tool analyzes PDB files and creates configuration files for binder design.
    It's fast and doesn't require GPU resources.

    The generated target_settings.json follows the format expected by run_bindcraft.py:
    ```json
    {
        "design_path": "/path/to/output",
        "binder_name": "Binder",
        "starting_pdb": "/path/to/target.pdb",
        "chains": "A",
        "target_hotspot_residues": "18,30,42",
        "lengths": [130, 130],
        "number_of_final_designs": 1
    }
    ```

    Input: Path to target PDB file
    Output: Dictionary with generated configuration and validation results
    """
    logger.info(f"generate_config called with input={input_file}")

    from use_case_5_config_generator import run_config_generator

    try:
        result = run_config_generator(
            input_file=input_file,
            output_file=output_file,
            chains=chains,
            binder_length=binder_length,
            validate_config=validate,
            analysis_type=analysis_type,
            num_designs=num_designs,
            name=binder_name,
            hotspot=hotspot
        )

        if result.get("success"):
            logger.info(f"Config generated successfully: {result.get('output_dir')}")
            return {"status": "success", **result}
        else:
            logger.error(f"Config generation failed: {result.get('error')}")
            return {"status": "error", "error": result.get("error", "Unknown error"), **result}

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.exception(f"Config generation failed: {e}")
        return {"status": "error", "error": f"Config generation failed: {str(e)}"}


@bindcraft_config_mcp.tool
def validate_config(
    settings_file: Annotated[str, "Path to target settings JSON (required)"],
    filters_file: Annotated[Optional[str], "Path to filter settings JSON"] = None,
    advanced_file: Annotated[Optional[str], "Path to advanced settings JSON"] = None,
    check_file_exists: Annotated[bool, "Whether to verify referenced files exist"] = True
) -> dict:
    """
    Validate BindCraft configuration files before job submission.

    Use this tool to check configuration files are correct before submitting
    design jobs. This helps catch errors early and avoid wasted compute time.

    Validates:
    - Required fields are present (design_path, binder_name, starting_pdb, etc.)
    - Field types are correct (lengths must be [min, max] array, etc.)
    - Referenced files exist (target PDB, etc.)
    - Filter settings have correct structure (threshold, higher keys)
    - Advanced settings have valid values

    Example usage:
        # Validate before submitting a job
        result = validate_config(
            settings_file="target_settings.json",
            filters_file="filters.json",
            advanced_file="advanced.json"
        )
        if result["valid"]:
            # Safe to submit job
            bindcraft_design_binder(settings_json="target_settings.json")

    Input: Path to target settings JSON file (required), optional filter and advanced settings
    Output: Dictionary with validation results including errors, warnings, and summary
    """
    logger.info(f"validate_config called with settings={settings_file}")

    from lib.config_validation import validate_all_configs

    try:
        result = validate_all_configs(
            target_settings=settings_file,
            filters_settings=filters_file,
            advanced_settings=advanced_file,
            check_file_exists=check_file_exists
        )

        # Format response
        response = {
            "status": "success" if result["valid"] else "error",
            "valid": result["valid"],
            "errors": result["errors"],
            "warnings": result["warnings"],
            "summary": result["summary"]
        }

        # Add target config info if available
        if result.get("target_validation") and result["target_validation"].get("config"):
            target_config = result["target_validation"]["config"]
            response["target_info"] = {
                "design_path": target_config.get("design_path"),
                "binder_name": target_config.get("binder_name"),
                "starting_pdb": target_config.get("starting_pdb"),
                "chains": target_config.get("chains"),
                "lengths": target_config.get("lengths"),
                "number_of_final_designs": target_config.get("number_of_final_designs")
            }

        if result["valid"]:
            logger.info("Configuration validation passed")
        else:
            logger.warning(f"Configuration validation failed with {len(result['errors'])} errors")

        return response

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return {
            "status": "error",
            "valid": False,
            "errors": [str(e)],
            "warnings": [],
            "summary": {"target_valid": False}
        }
    except Exception as e:
        logger.exception(f"Validation failed: {e}")
        return {
            "status": "error",
            "valid": False,
            "errors": [f"Validation failed: {str(e)}"],
            "warnings": [],
            "summary": {"target_valid": False}
        }
