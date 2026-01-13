#!/usr/bin/env python3
"""Test script for config generation and validation workflow."""

import sys
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR / "clean_scripts"))
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from use_case_5_config_generator import run_config_generator
from lib.config_validation import validate_all_configs, validate_target_settings


def test_generate_and_validate():
    """Test generating configs and then validating them."""

    # You can use any PDB file here - update the path as needed
    test_pdb = SCRIPT_DIR / "examples" / "data" / "PDL1.pdb"
    output_dir = SCRIPT_DIR / "test_output" / "validation_test"

    print("=" * 60)
    print("Step 1: Generate Configuration")
    print("=" * 60)

    # Check if test PDB exists
    if not test_pdb.exists():
        print(f"Test PDB not found at: {test_pdb}")
        print("Please update the test_pdb path to point to an existing PDB file.")
        # Try to find any PDB in examples/data
        examples_dir = SCRIPT_DIR / "examples" / "data"
        if examples_dir.exists():
            pdbs = list(examples_dir.glob("*.pdb"))
            if pdbs:
                test_pdb = pdbs[0]
                print(f"Found alternative PDB: {test_pdb}")
            else:
                print("No PDB files found in examples/data/")
                return
        else:
            print(f"Examples directory not found: {examples_dir}")
            return

    print(f"Input PDB: {test_pdb}")
    print(f"Output dir: {output_dir}")

    # Generate config
    gen_result = run_config_generator(
        input_file=str(test_pdb),
        output_file=str(output_dir),
        chains="A",
        binder_length=100,
        num_designs=1,
        name="TestBinder",
        validate_config=True  # Internal validation
    )

    if gen_result.get("success"):
        print(f"\n✅ Config generation successful!")
        print(f"   Output directory: {gen_result['output_dir']}")
        print(f"   Files generated: {gen_result['result']['files_generated']}")
    else:
        print(f"\n❌ Config generation failed: {gen_result.get('error')}")
        return

    print("\n" + "=" * 60)
    print("Step 2: Validate Generated Configuration")
    print("=" * 60)

    # Get paths to generated files
    config_dir = Path(gen_result['output_dir'])
    target_file = config_dir / "target_settings.json"
    filters_file = config_dir / "default_filters.json"
    advanced_file = config_dir / "default_4stage_multimer.json"

    print(f"\nValidating files:")
    print(f"   Target: {target_file}")
    print(f"   Filters: {filters_file}")
    print(f"   Advanced: {advanced_file}")

    # Validate using the validation module
    validation_result = validate_all_configs(
        target_settings=str(target_file),
        filters_settings=str(filters_file) if filters_file.exists() else None,
        advanced_settings=str(advanced_file) if advanced_file.exists() else None,
        check_file_exists=True
    )

    print(f"\n{'=' * 40}")
    print("Validation Results:")
    print(f"{'=' * 40}")
    print(f"Overall Valid: {validation_result['valid']}")

    if validation_result['errors']:
        print(f"\n❌ Errors ({len(validation_result['errors'])}):")
        for error in validation_result['errors']:
            print(f"   - {error}")

    if validation_result['warnings']:
        print(f"\n⚠️  Warnings ({len(validation_result['warnings'])}):")
        for warning in validation_result['warnings']:
            print(f"   - {warning}")

    print(f"\nSummary: {validation_result['summary']}")

    if validation_result['valid']:
        print("\n✅ All configurations are valid and ready for job submission!")
    else:
        print("\n❌ Configuration validation failed - fix errors before submitting jobs.")

    return validation_result


def test_validate_target_only():
    """Test validating just the target settings."""
    print("\n" + "=" * 60)
    print("Test: Validate Target Settings Only")
    print("=" * 60)

    # Create a sample target settings dict to validate
    sample_config = {
        "design_path": "/tmp/test_output",
        "binder_name": "TestBinder",
        "starting_pdb": "/nonexistent/path.pdb",  # Will trigger warning
        "chains": "A",
        "lengths": [100, 120],
        "number_of_final_designs": 3
    }

    result = validate_target_settings(sample_config, check_file_exists=True)

    print(f"Valid: {result['valid']}")
    print(f"Errors: {result['errors']}")
    print(f"Warnings: {result['warnings']}")

    return result


def test_invalid_config():
    """Test validation catches invalid configs."""
    print("\n" + "=" * 60)
    print("Test: Invalid Configuration Detection")
    print("=" * 60)

    # Missing required fields
    invalid_config = {
        "binder_name": "TestBinder",
        # Missing: design_path, starting_pdb, chains, lengths, number_of_final_designs
    }

    result = validate_target_settings(invalid_config, check_file_exists=False)

    print(f"Valid: {result['valid']} (expected: False)")
    print(f"Errors detected: {len(result['errors'])}")
    for error in result['errors']:
        print(f"   - {error}")

    # Test invalid lengths format
    invalid_lengths = {
        "design_path": "/tmp/test",
        "binder_name": "Test",
        "starting_pdb": "/tmp/test.pdb",
        "chains": "A",
        "lengths": 100,  # Should be [min, max] array!
        "number_of_final_designs": 1
    }

    result2 = validate_target_settings(invalid_lengths, check_file_exists=False)
    print(f"\nLengths format test - Valid: {result2['valid']} (expected: False)")
    for error in result2['errors']:
        print(f"   - {error}")

    return result, result2


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BindCraft Config Validation Test Suite")
    print("=" * 60 + "\n")

    # Run tests
    test_invalid_config()
    test_validate_target_only()
    test_generate_and_validate()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
