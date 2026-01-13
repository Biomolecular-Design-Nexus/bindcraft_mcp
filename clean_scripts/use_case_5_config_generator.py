#!/usr/bin/env python3
"""
Script: use_case_5_config_generator.py
Description: Generate BindCraft configuration files from PDB structures

Original Use Case: examples/use_case_5_config_generator.py
Dependencies Removed: Inlined MCP decorators, made BioPython optional

Usage:
    python clean_scripts/use_case_5_config_generator.py --input <pdb_file> --output <output_dir>

Example:
    python clean_scripts/use_case_5_config_generator.py --input examples/data/PDL1.pdb --output results/configs
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import json
import sys
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

# Add lib to path for shared utilities
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from lib.io import load_json, save_json, resolve_path, ensure_directory, copy_file
from lib.bindcraft import get_default_settings_paths

# Optional BioPython import
try:
    from Bio import PDB
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG_GEN = {
    "name": "ConfigBinder",
    "chains": "A",
    "binder_length": 130,
    "num_designs": 1,
    "auto_suggest_hotspots": True,
    "validate_config": True,
    "copy_defaults": True
}

# ==============================================================================
# Core Functions (main logic extracted from use case)
# ==============================================================================
def analyze_pdb_structure(pdb_path: Path) -> Dict[str, Any]:
    """Analyze PDB structure to extract useful information."""
    info = {
        "chains": [],
        "total_residues": 0,
        "chain_lengths": {},
        "chain_types": {},
        "suggested_hotspots": [],
        "warnings": []
    }

    if not HAS_BIOPYTHON:
        info["warnings"].append("BioPython not available - using basic analysis")
        return basic_pdb_analysis(pdb_path, info)

    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("target", pdb_path)

        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                residues = list(chain.get_residues())

                # Only count standard amino acids
                aa_residues = [r for r in residues if r.get_id()[0] == ' ']

                info["chains"].append(chain_id)
                info["chain_lengths"][chain_id] = len(aa_residues)
                info["total_residues"] += len(aa_residues)
                info["chain_types"][chain_id] = "protein"  # Simplified

                # Suggest hotspot residues (every 12th residue as a simple heuristic)
                if aa_residues:
                    hotspots = []
                    for i in range(0, len(aa_residues), 12):
                        if i + 1 <= len(aa_residues):
                            hotspots.append(str(aa_residues[i].get_id()[1]))

                    if hotspots and chain_id == info["chains"][0]:  # Only for first chain
                        info["suggested_hotspots"] = hotspots[:3]  # Limit to 3

            break  # Only process first model

    except Exception as e:
        info["warnings"].append(f"PDB parsing error: {str(e)}")
        info = basic_pdb_analysis(pdb_path, info)

    return info


def basic_pdb_analysis(pdb_path: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    """Basic PDB analysis without BioPython."""
    try:
        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        chains_found = set()
        residue_count = 0

        for line in lines:
            if line.startswith('ATOM') and len(line) > 21:
                chain_id = line[21].strip()
                if chain_id:
                    chains_found.add(chain_id)

                    # Count unique residues (simplified)
                    try:
                        res_num = int(line[22:26].strip())
                        if res_num > residue_count:
                            residue_count = res_num
                    except ValueError:
                        pass

        info["chains"] = sorted(list(chains_found))
        info["total_residues"] = residue_count

        # Basic hotspot suggestion (every 20th residue)
        if residue_count > 0:
            hotspots = []
            for i in range(20, min(residue_count + 1, 100), 20):
                hotspots.append(str(i))
            info["suggested_hotspots"] = hotspots[:3]

        for chain in info["chains"]:
            info["chain_lengths"][chain] = residue_count // len(info["chains"])
            info["chain_types"][chain] = "protein"

    except Exception as e:
        info["warnings"].append(f"Basic PDB analysis error: {str(e)}")

    return info


def generate_target_settings(
    target_pdb: str,
    output_dir: str,
    name: str = "ConfigBinder",
    chains: str = "A",
    hotspot: Optional[str] = None,
    binder_length: int = 130,
    num_designs: int = 1,
    pdb_analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate target settings JSON for BindCraft.

    Returns settings in the format expected by run_bindcraft.py:
    - design_path: Output directory for designed binders
    - binder_name: Prefix name for designed binders
    - starting_pdb: Target protein PDB structure
    - chains: Target chain(s) to design binder against
    - target_hotspot_residues: Residue numbers to target (comma-separated)
    - lengths: [min_length, max_length] range for binder length
    - number_of_final_designs: Target number of accepted designs
    """
    target_pdb_abs = resolve_path(target_pdb)
    output_dir_abs = resolve_path(output_dir)

    # Determine hotspot residues
    hotspot_residues = ""
    if hotspot:
        hotspot_residues = hotspot
    elif pdb_analysis and pdb_analysis.get("suggested_hotspots"):
        hotspot_residues = ",".join(pdb_analysis["suggested_hotspots"])

    # Use correct field names matching run_bindcraft.py expectations
    settings = {
        "design_path": output_dir_abs,                    # NOT "output_dir"
        "binder_name": name,
        "starting_pdb": target_pdb_abs,                   # NOT "target_pdb"
        "chains": chains,                                  # NOT "target_chains"
        "target_hotspot_residues": hotspot_residues,      # NOT "hotspot"
        "lengths": [binder_length, binder_length],        # Array format, NOT single int
        "number_of_final_designs": num_designs            # NOT "num_designs"
    }

    # Add metadata if analysis available (prefixed with _ to mark as non-essential)
    if pdb_analysis:
        settings["_metadata"] = {
            "pdb_analysis": pdb_analysis,
            "generated_by": "config_generator_script"
        }

    return settings


def validate_configuration(config_dir: Path) -> Dict[str, Any]:
    """Validate generated configuration files.

    Validates that settings match the schema expected by run_bindcraft.py.
    """
    validation_result = {
        "valid": True,
        "files_checked": [],
        "errors": [],
        "warnings": []
    }

    # Check target settings
    target_file = config_dir / "target_settings.json"
    if target_file.exists():
        try:
            settings = load_json(target_file)
            validation_result["files_checked"].append("target_settings.json")

            # Required fields for run_bindcraft.py
            required_fields = [
                "design_path",              # Output directory
                "binder_name",              # Name prefix for designs
                "starting_pdb",             # Target PDB file
                "chains",                   # Target chains
                "lengths",                  # [min, max] binder length
                "number_of_final_designs"   # Target number of designs
            ]

            for field in required_fields:
                if field not in settings:
                    validation_result["errors"].append(f"Missing required field: {field}")
                    validation_result["valid"] = False

            # Validate 'lengths' format (must be array)
            if "lengths" in settings:
                if not isinstance(settings["lengths"], list):
                    validation_result["errors"].append(
                        "'lengths' must be an array [min, max], not a single integer"
                    )
                    validation_result["valid"] = False
                elif len(settings["lengths"]) != 2:
                    validation_result["errors"].append(
                        "'lengths' must have exactly 2 elements: [min_length, max_length]"
                    )
                    validation_result["valid"] = False

            # Validate 'starting_pdb' exists
            if "starting_pdb" in settings:
                pdb_path = Path(settings["starting_pdb"])
                if not pdb_path.exists():
                    validation_result["warnings"].append(
                        f"Target PDB file not found: {settings['starting_pdb']}"
                    )

        except Exception as e:
            validation_result["errors"].append(f"Error reading target_settings.json: {str(e)}")
            validation_result["valid"] = False

    # Check filter settings
    filter_file = config_dir / "default_filters.json"
    if filter_file.exists():
        try:
            load_json(filter_file)
            validation_result["files_checked"].append("default_filters.json")
        except Exception as e:
            validation_result["errors"].append(f"Error reading default_filters.json: {str(e)}")
            validation_result["valid"] = False

    # Check advanced settings
    advanced_file = config_dir / "default_4stage_multimer.json"
    if advanced_file.exists():
        try:
            load_json(advanced_file)
            validation_result["files_checked"].append("default_4stage_multimer.json")
        except Exception as e:
            validation_result["errors"].append(f"Error reading default_4stage_multimer.json: {str(e)}")
            validation_result["valid"] = False

    return validation_result


def run_config_generator(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for generating BindCraft configuration files.

    Args:
        input_file: Path to target PDB file
        output_file: Path to output directory (optional)
        config: Configuration dict (uses DEFAULT_CONFIG_GEN if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - result: Configuration generation results
            - output_dir: Path to configuration directory
            - metadata: Execution metadata

    Example:
        >>> result = run_config_generator("target.pdb", "configs/")
        >>> print(f"Configs generated: {result['result']['files_generated']}")
    """
    # Setup
    input_file = Path(input_file)
    final_config = {**DEFAULT_CONFIG_GEN, **(config or {}), **kwargs}

    if not input_file.exists():
        raise FileNotFoundError(f"Input PDB file not found: {input_file}")

    # Determine output directory
    if output_file:
        output_dir = Path(output_file)
    else:
        output_dir = Path(f"results/{final_config['name']}_configs")

    output_dir = ensure_directory(output_dir)

    try:
        print(f"🧬 Analyzing PDB structure...")
        print(f"   Input: {input_file}")
        print(f"   Output: {output_dir}")

        # Analyze PDB structure
        pdb_analysis = None
        if final_config["auto_suggest_hotspots"]:
            pdb_analysis = analyze_pdb_structure(input_file)

            if pdb_analysis.get("warnings"):
                for warning in pdb_analysis["warnings"]:
                    print(f"   Warning: {warning}")

            if pdb_analysis.get("chains"):
                print(f"   Chains found: {', '.join(pdb_analysis['chains'])}")
                print(f"   Total residues: {pdb_analysis['total_residues']}")

                if pdb_analysis.get("suggested_hotspots"):
                    print(f"   Suggested hotspots: {', '.join(pdb_analysis['suggested_hotspots'])}")

        # Generate target settings
        target_settings = generate_target_settings(
            target_pdb=str(input_file),
            output_dir=str(output_dir / "job_output"),  # Subdirectory for actual job output
            name=final_config["name"],
            chains=final_config["chains"],
            hotspot=final_config.get("hotspot"),
            binder_length=final_config["binder_length"],
            num_designs=final_config["num_designs"],
            pdb_analysis=pdb_analysis
        )

        # Save target settings
        target_file = output_dir / "target_settings.json"
        save_json(target_settings, target_file)
        print(f"   Generated: target_settings.json")

        files_generated = ["target_settings.json"]

        # Copy default configuration files if requested
        if final_config["copy_defaults"]:
            try:
                defaults = get_default_settings_paths()

                # Copy filters
                filter_src = Path(defaults["filters"])
                filter_dst = output_dir / "default_filters.json"
                if filter_src.exists():
                    copy_file(filter_src, filter_dst)
                    files_generated.append("default_filters.json")
                    print(f"   Copied: default_filters.json")

                # Copy advanced settings
                advanced_src = Path(defaults["advanced"])
                advanced_dst = output_dir / "default_4stage_multimer.json"
                if advanced_src.exists():
                    copy_file(advanced_src, advanced_dst)
                    files_generated.append("default_4stage_multimer.json")
                    print(f"   Copied: default_4stage_multimer.json")

            except Exception as e:
                print(f"   Warning: Could not copy default files: {str(e)}")

        # Generate usage README
        readme_content = f"""# BindCraft Configuration Files

Generated from: {input_file.name}
Generated on: {json.dumps(target_settings.get('_metadata', {}).get('generated_by', 'Unknown'))}

## Files

- `target_settings.json`: Main target configuration
- `default_filters.json`: Filter settings for design quality
- `default_4stage_multimer.json`: Advanced pipeline settings

## Usage

### Quick Design (synchronous)
```bash
python clean_scripts/use_case_1_quick_design.py --input {input_file.name} --config target_settings.json --output results/
```

### Async Submission
```bash
python clean_scripts/use_case_2_async_submission.py --input {input_file.name} --config target_settings.json --output results/
```

### Batch Processing
```bash
python clean_scripts/use_case_4_batch_design.py --input {input_file.name} --config target_settings.json --output results/
```

## Configuration Summary

- **Target**: {final_config['name']}
- **Chains**: {final_config['chains']}
- **Designs**: {final_config['num_designs']}
- **Binder Length**: {final_config['binder_length']}"""

        if pdb_analysis and pdb_analysis.get("suggested_hotspots"):
            readme_content += f"\n- **Suggested Hotspots**: {', '.join(pdb_analysis['suggested_hotspots'])}"

        readme_file = output_dir / "README.md"
        readme_file.write_text(readme_content)
        files_generated.append("README.md")
        print(f"   Generated: README.md")

        # Validate configurations if requested
        validation_result = None
        if final_config["validate_config"]:
            validation_result = validate_configuration(output_dir)
            print(f"   Validation: {'✅ Passed' if validation_result['valid'] else '❌ Failed'}")

            if validation_result["errors"]:
                for error in validation_result["errors"]:
                    print(f"     Error: {error}")

        return {
            "success": True,
            "result": {
                "files_generated": files_generated,
                "target_settings": target_settings,
                "pdb_analysis": pdb_analysis,
                "validation": validation_result
            },
            "output_dir": str(output_dir),
            "target_file": str(target_file),
            "metadata": {
                "input_file": str(input_file),
                "config": final_config,
                "generation_time": "Configuration generated"
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Configuration generation failed: {str(e)}",
            "output_dir": str(output_dir) if 'output_dir' in locals() else None,
            "metadata": {
                "input_file": str(input_file),
                "config": final_config
            }
        }


# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', required=True, help='Target PDB file path')
    parser.add_argument('--output', '-o', help='Output directory path')
    parser.add_argument('--name', default='ConfigBinder', help='Binder name (default: ConfigBinder)')
    parser.add_argument('--chains', default='A', help='Target chains (default: A)')
    parser.add_argument('--hotspot', help='Hotspot residues (comma-separated)')
    parser.add_argument('--num-designs', type=int, default=1, help='Number of designs (default: 1)')
    parser.add_argument('--binder-length', type=int, default=130, help='Binder length (default: 130)')
    parser.add_argument('--config', '-c', help='Config file (JSON)')
    parser.add_argument('--validate', action='store_true', help='Validate generated configs')
    parser.add_argument('--no-defaults', action='store_true', help='Skip copying default files')
    parser.add_argument('--no-analysis', action='store_true', help='Skip PDB structure analysis')

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        config = load_json(args.config)

    # Override with CLI args
    config.update({
        "name": args.name,
        "chains": args.chains,
        "hotspot": args.hotspot,
        "num_designs": args.num_designs,
        "binder_length": args.binder_length,
        "validate_config": args.validate,
        "copy_defaults": not args.no_defaults,
        "auto_suggest_hotspots": not args.no_analysis
    })

    # Remove None values
    config = {k: v for k, v in config.items() if v is not None}

    # Run configuration generation
    result = run_config_generator(
        input_file=args.input,
        output_file=args.output,
        config=config
    )

    if result["success"]:
        generation_info = result["result"]
        print(f"\n✅ Configuration generated successfully!")
        print(f"   Output directory: {result['output_dir']}")
        print(f"   Files generated: {', '.join(generation_info['files_generated'])}")

        if generation_info.get("validation"):
            validation = generation_info["validation"]
            print(f"   Validation: {'✅ Passed' if validation['valid'] else '❌ Failed'}")

        if generation_info.get("pdb_analysis"):
            analysis = generation_info["pdb_analysis"]
            if analysis.get("suggested_hotspots"):
                print(f"   Suggested hotspots: {', '.join(analysis['suggested_hotspots'])}")

        print(f"\n📖 See README.md for usage instructions")
    else:
        print(f"\n❌ Configuration generation failed: {result['error']}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())