"""Configuration validation module for BindCraft.

This module provides comprehensive validation of BindCraft configuration files
before job submission to ensure configs are correct and will not cause runtime errors.

Usage:
    from lib.config_validation import validate_all_configs, validate_target_settings

    # Validate all configs at once
    result = validate_all_configs(
        target_settings="path/to/target.json",
        filters_settings="path/to/filters.json",
        advanced_settings="path/to/advanced.json"
    )

    # Validate individual config
    result = validate_target_settings("path/to/target.json")
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List


# ==============================================================================
# Required Fields Definition
# ==============================================================================

# Required fields for target_settings.json (used by run_bindcraft.py)
TARGET_REQUIRED_FIELDS = {
    "design_path": {
        "type": str,
        "description": "Output directory for designed binders"
    },
    "binder_name": {
        "type": str,
        "description": "Name prefix for designed binders"
    },
    "starting_pdb": {
        "type": str,
        "description": "Path to target PDB file",
        "check_exists": True
    },
    "chains": {
        "type": str,
        "description": "Target chain(s) to design binder against"
    },
    "lengths": {
        "type": list,
        "description": "Binder length range [min, max]",
        "length": 2
    },
    "number_of_final_designs": {
        "type": int,
        "description": "Number of final designs to generate"
    }
}

# Optional but commonly used target fields
TARGET_OPTIONAL_FIELDS = {
    "target_hotspot_residues": {
        "type": str,
        "description": "Comma-separated residue numbers to target"
    }
}

# Key advanced settings fields that affect execution
ADVANCED_KEY_FIELDS = {
    "design_algorithm": {
        "type": str,
        "description": "Design algorithm to use",
        "allowed_values": ["AfDesign", "RFdiffusion"]
    },
    "use_multimer_design": {
        "type": bool,
        "description": "Whether to use multimer models for design"
    },
    "num_recycles_design": {
        "type": int,
        "description": "Number of recycles during design"
    },
    "num_recycles_validation": {
        "type": int,
        "description": "Number of recycles during validation"
    },
    "enable_mpnn": {
        "type": bool,
        "description": "Whether to enable MPNN sequence optimization"
    },
    "num_seqs": {
        "type": int,
        "description": "Number of MPNN sequences to generate"
    }
}

# Key filter settings that affect design acceptance
FILTER_KEY_FIELDS = {
    "Average_pLDDT": {
        "type": dict,
        "description": "pLDDT threshold filter"
    },
    "Average_i_pTM": {
        "type": dict,
        "description": "Interface pTM threshold filter"
    }
}


# ==============================================================================
# Validation Functions
# ==============================================================================

def load_config(config: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Load configuration from file path or return dict directly."""
    if isinstance(config, dict):
        return config

    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


def validate_target_settings(
    config: Union[str, Path, Dict[str, Any]],
    check_file_exists: bool = True
) -> Dict[str, Any]:
    """
    Validate target settings configuration.

    Args:
        config: Path to target settings JSON or dict
        check_file_exists: Whether to check if referenced files exist

    Returns:
        Dict with validation results:
            - valid: bool indicating if config is valid
            - errors: list of error messages
            - warnings: list of warning messages
            - config: the loaded configuration (if valid)
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "config": None,
        "config_type": "target_settings"
    }

    try:
        settings = load_config(config)
        result["config"] = settings
    except FileNotFoundError as e:
        result["valid"] = False
        result["errors"].append(str(e))
        return result
    except json.JSONDecodeError as e:
        result["valid"] = False
        result["errors"].append(f"Invalid JSON format: {e}")
        return result
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Failed to load config: {e}")
        return result

    # Check required fields
    for field, spec in TARGET_REQUIRED_FIELDS.items():
        if field not in settings:
            result["valid"] = False
            result["errors"].append(f"Missing required field '{field}': {spec['description']}")
            continue

        value = settings[field]

        # Type check
        if not isinstance(value, spec["type"]):
            result["valid"] = False
            result["errors"].append(
                f"Field '{field}' must be {spec['type'].__name__}, got {type(value).__name__}"
            )
            continue

        # Length check for lists
        if spec["type"] == list and "length" in spec:
            if len(value) != spec["length"]:
                result["valid"] = False
                result["errors"].append(
                    f"Field '{field}' must have {spec['length']} elements, got {len(value)}"
                )

        # File existence check
        if check_file_exists and spec.get("check_exists"):
            file_path = Path(value)
            if not file_path.exists():
                result["warnings"].append(f"File not found: {value} (field: {field})")

    # Validate lengths values are positive integers
    if "lengths" in settings and isinstance(settings["lengths"], list):
        lengths = settings["lengths"]
        if len(lengths) == 2:
            min_len, max_len = lengths
            if not (isinstance(min_len, int) and isinstance(max_len, int)):
                result["valid"] = False
                result["errors"].append("'lengths' values must be integers")
            elif min_len > max_len:
                result["valid"] = False
                result["errors"].append(f"'lengths' min ({min_len}) cannot be greater than max ({max_len})")
            elif min_len <= 0:
                result["valid"] = False
                result["errors"].append("'lengths' values must be positive")

    # Validate number_of_final_designs
    if "number_of_final_designs" in settings:
        num = settings["number_of_final_designs"]
        if isinstance(num, int) and num <= 0:
            result["valid"] = False
            result["errors"].append("'number_of_final_designs' must be a positive integer")

    # Validate chains format
    if "chains" in settings:
        chains = settings["chains"]
        if isinstance(chains, str) and chains:
            # Check for valid chain identifiers (letters A-Z or comma-separated)
            chain_list = [c.strip() for c in chains.split(',')]
            for chain in chain_list:
                if not chain.isalpha() or len(chain) != 1:
                    result["warnings"].append(
                        f"Unusual chain identifier: '{chain}'. Expected single letters (A, B, C, etc.)"
                    )

    # Check optional fields
    for field, spec in TARGET_OPTIONAL_FIELDS.items():
        if field in settings:
            value = settings[field]
            if not isinstance(value, spec["type"]):
                result["warnings"].append(
                    f"Optional field '{field}' should be {spec['type'].__name__}, got {type(value).__name__}"
                )

    return result


def validate_filter_settings(
    config: Union[str, Path, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate filter settings configuration.

    Args:
        config: Path to filter settings JSON or dict

    Returns:
        Dict with validation results
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "config": None,
        "config_type": "filter_settings"
    }

    try:
        settings = load_config(config)
        result["config"] = settings
    except FileNotFoundError as e:
        result["valid"] = False
        result["errors"].append(str(e))
        return result
    except json.JSONDecodeError as e:
        result["valid"] = False
        result["errors"].append(f"Invalid JSON format: {e}")
        return result
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Failed to load config: {e}")
        return result

    # Validate filter structure - each filter should have 'threshold' and 'higher' keys
    for filter_name, filter_config in settings.items():
        if not isinstance(filter_config, dict):
            result["warnings"].append(f"Filter '{filter_name}' should be a dict")
            continue

        # Special case for InterfaceAAs filters (nested structure)
        if "InterfaceAAs" in filter_name:
            for aa, aa_config in filter_config.items():
                if isinstance(aa_config, dict):
                    if "threshold" not in aa_config:
                        result["warnings"].append(
                            f"Filter '{filter_name}.{aa}' missing 'threshold' key"
                        )
                    if "higher" not in aa_config:
                        result["warnings"].append(
                            f"Filter '{filter_name}.{aa}' missing 'higher' key"
                        )
        else:
            # Standard filter validation
            if "threshold" not in filter_config:
                result["warnings"].append(f"Filter '{filter_name}' missing 'threshold' key")
            if "higher" not in filter_config:
                result["warnings"].append(f"Filter '{filter_name}' missing 'higher' key")

            # Validate threshold type
            if "threshold" in filter_config:
                threshold = filter_config["threshold"]
                if threshold is not None and not isinstance(threshold, (int, float)):
                    result["errors"].append(
                        f"Filter '{filter_name}' threshold must be numeric or null"
                    )
                    result["valid"] = False

            # Validate higher type
            if "higher" in filter_config:
                higher = filter_config["higher"]
                if not isinstance(higher, bool):
                    result["errors"].append(
                        f"Filter '{filter_name}' 'higher' must be boolean"
                    )
                    result["valid"] = False

    return result


def validate_advanced_settings(
    config: Union[str, Path, Dict[str, Any]],
    check_file_exists: bool = True
) -> Dict[str, Any]:
    """
    Validate advanced settings configuration.

    Args:
        config: Path to advanced settings JSON or dict
        check_file_exists: Whether to check if referenced files exist

    Returns:
        Dict with validation results
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "config": None,
        "config_type": "advanced_settings"
    }

    try:
        settings = load_config(config)
        result["config"] = settings
    except FileNotFoundError as e:
        result["valid"] = False
        result["errors"].append(str(e))
        return result
    except json.JSONDecodeError as e:
        result["valid"] = False
        result["errors"].append(f"Invalid JSON format: {e}")
        return result
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Failed to load config: {e}")
        return result

    # Check key fields
    for field, spec in ADVANCED_KEY_FIELDS.items():
        if field in settings:
            value = settings[field]

            # Type check
            if not isinstance(value, spec["type"]):
                result["warnings"].append(
                    f"Field '{field}' should be {spec['type'].__name__}, got {type(value).__name__}"
                )

            # Allowed values check
            if "allowed_values" in spec and value not in spec["allowed_values"]:
                result["warnings"].append(
                    f"Field '{field}' value '{value}' not in allowed values: {spec['allowed_values']}"
                )

    # Check file paths if they exist
    file_path_fields = ["af_params_dir", "dssp_path", "dalphaball_path"]
    for field in file_path_fields:
        if field in settings and settings[field]:
            path = Path(settings[field])
            if check_file_exists and not path.exists():
                # Only warn, don't error - paths might be set at runtime
                result["warnings"].append(f"Path not found: {settings[field]} (field: {field})")

    # Validate numeric ranges
    numeric_positive_fields = [
        "num_recycles_design", "num_recycles_validation", "num_seqs",
        "max_mpnn_sequences", "max_trajectories", "soft_iterations",
        "hard_iterations", "greedy_iterations"
    ]
    for field in numeric_positive_fields:
        if field in settings:
            value = settings[field]
            if isinstance(value, (int, float)) and value < 0:
                result["warnings"].append(f"Field '{field}' should be non-negative, got {value}")

    return result


def validate_all_configs(
    target_settings: Union[str, Path, Dict[str, Any]],
    filters_settings: Optional[Union[str, Path, Dict[str, Any]]] = None,
    advanced_settings: Optional[Union[str, Path, Dict[str, Any]]] = None,
    check_file_exists: bool = True
) -> Dict[str, Any]:
    """
    Validate all BindCraft configuration files.

    This is the main validation function to use before submitting jobs.

    Args:
        target_settings: Path to target settings JSON or dict (required)
        filters_settings: Path to filter settings JSON or dict (optional)
        advanced_settings: Path to advanced settings JSON or dict (optional)
        check_file_exists: Whether to verify referenced files exist

    Returns:
        Dict with comprehensive validation results:
            - valid: bool indicating if all configs are valid
            - errors: list of all error messages
            - warnings: list of all warning messages
            - target_validation: target settings validation result
            - filters_validation: filter settings validation result (if provided)
            - advanced_validation: advanced settings validation result (if provided)

    Example:
        >>> result = validate_all_configs(
        ...     target_settings="target.json",
        ...     filters_settings="filters.json",
        ...     advanced_settings="advanced.json"
        ... )
        >>> if result["valid"]:
        ...     print("All configs valid, ready to submit job")
        >>> else:
        ...     print(f"Errors: {result['errors']}")
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "target_validation": None,
        "filters_validation": None,
        "advanced_validation": None,
        "summary": {}
    }

    # Validate target settings (required)
    target_result = validate_target_settings(target_settings, check_file_exists)
    result["target_validation"] = target_result
    if not target_result["valid"]:
        result["valid"] = False
        result["errors"].extend([f"[target] {e}" for e in target_result["errors"]])
    result["warnings"].extend([f"[target] {w}" for w in target_result["warnings"]])

    # Validate filter settings (optional)
    if filters_settings is not None:
        filters_result = validate_filter_settings(filters_settings)
        result["filters_validation"] = filters_result
        if not filters_result["valid"]:
            result["valid"] = False
            result["errors"].extend([f"[filters] {e}" for e in filters_result["errors"]])
        result["warnings"].extend([f"[filters] {w}" for w in filters_result["warnings"]])

    # Validate advanced settings (optional)
    if advanced_settings is not None:
        advanced_result = validate_advanced_settings(advanced_settings, check_file_exists)
        result["advanced_validation"] = advanced_result
        if not advanced_result["valid"]:
            result["valid"] = False
            result["errors"].extend([f"[advanced] {e}" for e in advanced_result["errors"]])
        result["warnings"].extend([f"[advanced] {w}" for w in advanced_result["warnings"]])

    # Build summary
    result["summary"] = {
        "target_valid": target_result["valid"],
        "filters_valid": filters_result["valid"] if filters_settings else None,
        "advanced_valid": advanced_result["valid"] if advanced_settings else None,
        "total_errors": len(result["errors"]),
        "total_warnings": len(result["warnings"])
    }

    return result


# ==============================================================================
# CLI Interface for standalone validation
# ==============================================================================

def main():
    """Command-line interface for config validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate BindCraft configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate target settings only
    python config_validation.py --target target_settings.json

    # Validate all config files
    python config_validation.py --target target.json --filters filters.json --advanced advanced.json

    # Skip file existence checks
    python config_validation.py --target target.json --no-file-check
        """
    )

    parser.add_argument(
        "--target", "-t",
        required=True,
        help="Path to target settings JSON file"
    )
    parser.add_argument(
        "--filters", "-f",
        help="Path to filter settings JSON file"
    )
    parser.add_argument(
        "--advanced", "-a",
        help="Path to advanced settings JSON file"
    )
    parser.add_argument(
        "--no-file-check",
        action="store_true",
        help="Skip checking if referenced files exist"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    result = validate_all_configs(
        target_settings=args.target,
        filters_settings=args.filters,
        advanced_settings=args.advanced,
        check_file_exists=not args.no_file_check
    )

    if args.json:
        # Remove config objects for cleaner JSON output
        output = {k: v for k, v in result.items() if k not in ["target_validation", "filters_validation", "advanced_validation"]}
        output["summary"] = result["summary"]
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        print("\n" + "=" * 60)
        print("BindCraft Configuration Validation")
        print("=" * 60)

        status = "VALID" if result["valid"] else "INVALID"
        status_symbol = "\u2705" if result["valid"] else "\u274c"
        print(f"\nOverall Status: {status_symbol} {status}")

        if result["errors"]:
            print(f"\n\u274c Errors ({len(result['errors'])}):")
            for error in result["errors"]:
                print(f"   - {error}")

        if result["warnings"]:
            print(f"\n\u26a0\ufe0f  Warnings ({len(result['warnings'])}):")
            for warning in result["warnings"]:
                print(f"   - {warning}")

        print("\n" + "-" * 60)
        print("Summary:")
        print(f"   Target settings: {'Valid' if result['summary']['target_valid'] else 'Invalid'}")
        if result['summary']['filters_valid'] is not None:
            print(f"   Filter settings: {'Valid' if result['summary']['filters_valid'] else 'Invalid'}")
        if result['summary']['advanced_valid'] is not None:
            print(f"   Advanced settings: {'Valid' if result['summary']['advanced_valid'] else 'Invalid'}")
        print("=" * 60 + "\n")

    return 0 if result["valid"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
