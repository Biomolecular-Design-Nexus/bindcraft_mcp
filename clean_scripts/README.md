# MCP Scripts

Clean, self-contained scripts extracted from use cases for MCP tool wrapping.

## Design Principles

1. **Minimal Dependencies**: Only essential packages imported
2. **Self-Contained**: Functions inlined where possible
3. **Configurable**: Parameters in config files, not hardcoded
4. **MCP-Ready**: Each script has a main function ready for MCP wrapping

## Scripts

| Script | Description | Repo Dependent | Config |
|--------|-------------|----------------|--------|
| `use_case_1_quick_design.py` | Quick synchronous binder design | Yes (BindCraft scripts) | `configs/use_case_1.json` |
| `use_case_2_async_submission.py` | Async job submission | Yes (BindCraft scripts) | `configs/use_case_2.json` |
| `use_case_3_monitor_progress.py` | Monitor job progress | No | `configs/use_case_3.json` |
| `use_case_4_batch_design.py` | Batch processing | Yes (BindCraft scripts) | `configs/use_case_4.json` |
| `use_case_5_config_generator.py` | Generate configurations | No (BioPython optional) | `configs/use_case_5.json` |

## Dependencies Summary

### Essential Dependencies
- Python standard library (argparse, json, os, sys, pathlib, typing, subprocess, threading, time)
- No external packages required for basic functionality

### Optional Dependencies
- **BioPython**: For advanced PDB structure analysis (use_case_5)
- **loguru**: Enhanced logging (extracted to avoid dependency)

### Repo Dependencies
- **BindCraft scripts** (`scripts/run_bindcraft.py`): Required for use cases 1, 2, 4
- **Default configuration files** (`examples/data/*.json`): Used by all design scripts

## Usage

### Environment Setup
```bash
# Activate environment (prefer mamba over conda)
mamba activate ./env  # or: conda activate ./env
```

### Quick Design (Synchronous)
```bash
python clean_scripts/use_case_1_quick_design.py --input examples/data/PDL1.pdb --output results/quick_design
```

### Async Job Submission
```bash
python clean_scripts/use_case_2_async_submission.py --input examples/data/PDL1.pdb --output results/async_job
```

### Monitor Progress
```bash
python clean_scripts/use_case_3_monitor_progress.py --output results/async_job --detailed
```

### Batch Processing
```bash
python clean_scripts/use_case_4_batch_design.py --input examples/data/PDL1.pdb --output results/batch_jobs
```

### Configuration Generation
```bash
python clean_scripts/use_case_5_config_generator.py --input examples/data/PDL1.pdb --output results/configs --validate
```

### Using Custom Configurations
```bash
# Run with custom config file
python clean_scripts/use_case_1_quick_design.py --input FILE --output DIR --config configs/use_case_1_config.json

# Override specific parameters
python clean_scripts/use_case_1_quick_design.py --input FILE --output DIR --num-designs 3 --device 1
```

## Shared Library

Common functions are in `clean_scripts/lib/`:

### `lib/io.py`
- `load_json()`: Load JSON configuration files
- `save_json()`: Save data to JSON files
- `copy_file()`: Copy files with directory creation
- `resolve_path()`: Convert relative to absolute paths
- `ensure_directory()`: Create directory if it doesn't exist

### `lib/bindcraft.py`
- `get_bindcraft_path()`: Locate BindCraft scripts
- `get_default_settings_paths()`: Get default config file paths
- `run_command()`: Execute BindCraft with proper logging and GPU setup
- Configuration constants and path management

## For MCP Wrapping (Step 6)

Each script exports a main function that can be wrapped:

```python
from clean_scripts.use_case_1_quick_design import run_quick_design
from clean_scripts.use_case_2_async_submission import run_async_submission
from clean_scripts.use_case_3_monitor_progress import run_monitor_progress
from clean_scripts.use_case_4_batch_design import run_batch_design
from clean_scripts.use_case_5_config_generator import run_config_generator

# In MCP tool:
@mcp.tool()
def quick_design(input_file: str, output_file: str = None, **kwargs):
    """Design protein binders quickly (synchronous)."""
    return run_quick_design(input_file, output_file, **kwargs)

@mcp.tool()
def submit_async_job(input_file: str, output_file: str = None, **kwargs):
    """Submit binder design job asynchronously."""
    return run_async_submission(input_file, output_file, **kwargs)

@mcp.tool()
def monitor_progress(output_dir: str, **kwargs):
    """Monitor progress of running BindCraft jobs."""
    return run_monitor_progress(output_dir, **kwargs)

@mcp.tool()
def batch_process(input_file: str, output_file: str = None, **kwargs):
    """Process multiple targets in batch mode."""
    return run_batch_design(input_file, output_file, **kwargs)

@mcp.tool()
def generate_config(input_file: str, output_file: str = None, **kwargs):
    """Generate BindCraft configuration files from PDB."""
    return run_config_generator(input_file, output_file, **kwargs)
```

## Function Signatures

All main functions follow this pattern:

```python
def run_<use_case_name>(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Returns:
        Dict containing:
            - success: bool - Whether operation succeeded
            - result: Any - Main result data
            - output_dir: str - Path to output directory
            - metadata: Dict - Execution metadata
            - error: str - Error message (if success=False)
    """
```

## Error Handling

All scripts include comprehensive error handling:
- File not found errors
- BindCraft execution failures
- Configuration validation errors
- Process management errors

Errors are returned in the result dictionary rather than raising exceptions, making them suitable for MCP wrapping.

## Configuration Files

Each script can be configured using JSON files in `configs/`:
- `default_config.json`: Common defaults
- `use_case_X_config.json`: Use-case specific configurations

Configuration precedence (highest to lowest):
1. CLI arguments (`--num-designs 3`)
2. Custom config file (`--config myconfig.json`)
3. Default use-case config (`configs/use_case_X_config.json`)
4. Built-in defaults

## Performance Notes

- **GPU Utilization**: All design scripts support GPU selection via `--device`
- **Background Processing**: Use cases 2 and 4 run in detached background processes
- **Memory Efficiency**: Scripts load only required dependencies
- **Concurrent Jobs**: Batch processing supports multiple concurrent jobs (configurable)

## Troubleshooting

### Common Issues

1. **BindCraft scripts not found**
   ```
   FileNotFoundError: BindCraft scripts not found at /path/to/scripts
   ```
   **Solution**: Ensure `scripts/` directory exists with `run_bindcraft.py`

2. **GPU not available**
   ```
   CUDA_ERROR: No CUDA-capable device is detected
   ```
   **Solution**: Check GPU availability or use CPU mode (implementation dependent)

3. **Permission errors for background jobs**
   ```
   PermissionError: [Errno 13] Permission denied
   ```
   **Solution**: Check file permissions and directory access

### Debug Mode

All scripts support verbose output for debugging:
```bash
python clean_scripts/use_case_1_quick_design.py --input FILE --output DIR --verbose
```

### Log Files

Background jobs create log files:
- `{output_dir}/bindcraft_run.log`: BindCraft execution log
- `{output_dir}/target_settings.json`: Job configuration
- `{output_dir}/batch_jobs.json`: Batch processing info (use case 4)

## Testing

Verify scripts work correctly:
```bash
# Test configuration generation (no BindCraft dependency)
python clean_scripts/use_case_5_config_generator.py --input examples/data/PDL1.pdb --output test_config --validate

# Test monitoring (no BindCraft dependency)
python clean_scripts/use_case_3_monitor_progress.py --output results/nonexistent_job

# Test help for all scripts
for script in clean_scripts/use_case_*.py; do
    python "$script" --help
done
```