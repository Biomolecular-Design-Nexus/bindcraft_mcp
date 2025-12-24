# Configuration Files

JSON configuration files for MCP scripts, extracted from verified use cases.

## Files Overview

| Config File | Use Case | Description |
|------------|----------|-------------|
| `default_config.json` | All | Common defaults shared across scripts |
| `use_case_1_config.json` | Quick Design | Synchronous binder design configuration |
| `use_case_2_config.json` | Async Submission | Asynchronous job submission settings |
| `use_case_3_config.json` | Monitor Progress | Job monitoring parameters |
| `use_case_4_config.json` | Batch Design | Batch processing configuration |
| `use_case_5_config.json` | Config Generator | Configuration generation settings |

## Configuration Structure

### Common Parameters (in all configs)

```json
{
  "name": "Binder",          // Base name for generated binders
  "chains": "A",             // Target protein chains
  "hotspot": null,           // Hotspot residues (comma-separated)
  "binder_length": 130,      // Length of designed binder
  "num_designs": 1,          // Number of designs to generate
  "device": "0",             // GPU device number
  "filters_enabled": true    // Enable quality filters
}
```

### Processing Parameters

```json
{
  "processing": {
    "timeout": 3600,           // Maximum execution time (seconds)
    "synchronous": true,       // Wait for completion
    "asynchronous": false,     // Run in background
    "background_execution": false,  // Detach from parent process
    "log_file": "bindcraft_run.log"  // Log file name
  }
}
```

### Output Parameters

```json
{
  "output": {
    "save_statistics": true,   // Save design statistics
    "include_metadata": true,  // Include execution metadata
    "save_job_info": true,     // Save job information
    "include_pid": true        // Include process ID for background jobs
  }
}
```

## Using Configuration Files

### Method 1: Command Line
```bash
python clean_scripts/use_case_1_quick_design.py --config configs/use_case_1_config.json --input FILE --output DIR
```

### Method 2: Programmatically
```python
from clean_scripts.use_case_1_quick_design import run_quick_design
from clean_scripts.lib.io import load_json

config = load_json("configs/use_case_1_config.json")
result = run_quick_design("input.pdb", "output/", config=config)
```

### Method 3: Override Parameters
```bash
# Use config file but override specific parameters
python clean_scripts/use_case_1_quick_design.py \
    --config configs/use_case_1_config.json \
    --input FILE --output DIR \
    --num-designs 5 --device 1
```

## Configuration Precedence

Parameters are applied in this order (highest precedence first):

1. **Command line arguments**: `--num-designs 3`
2. **Explicitly provided config**: `--config custom.json`
3. **Function kwargs**: `run_quick_design(..., num_designs=3)`
4. **Built-in defaults**: Hardcoded in script

## Use Case Specific Details

### Use Case 1: Quick Design (`use_case_1_config.json`)

**Purpose**: Fast, synchronous binder design for testing and small-scale work

**Key Settings**:
- `num_designs: 1` - Single design for speed
- `synchronous: true` - Wait for completion
- `save_statistics: true` - Collect design metrics

**Typical Usage**:
```bash
python clean_scripts/use_case_1_quick_design.py \
    --config configs/use_case_1_config.json \
    --input target.pdb --output results/
```

### Use Case 2: Async Submission (`use_case_2_config.json`)

**Purpose**: Submit long-running jobs in the background

**Key Settings**:
- `num_designs: 3` - Multiple designs
- `asynchronous: true` - Background execution
- `include_pid: true` - Track process ID

**Typical Usage**:
```bash
python clean_scripts/use_case_2_async_submission.py \
    --config configs/use_case_2_config.json \
    --input target.pdb --output results/
```

### Use Case 3: Monitor Progress (`use_case_3_config.json`)

**Purpose**: Monitor running jobs

**Key Settings**:
- `refresh_interval: 30` - Check every 30 seconds
- `max_iterations: 120` - Run for max 1 hour
- `detailed: false` - Basic information

**Typical Usage**:
```bash
python clean_scripts/use_case_3_monitor_progress.py \
    --config configs/use_case_3_config.json \
    --output results/running_job/
```

### Use Case 4: Batch Design (`use_case_4_config.json`)

**Purpose**: Process multiple targets efficiently

**Key Settings**:
- `base_name: "BatchBinder"` - Base name for all jobs
- `num_designs: 2` - Moderate number per target
- `max_concurrent: 1` - One job at a time

**Typical Usage**:
```bash
python clean_scripts/use_case_4_batch_design.py \
    --config configs/use_case_4_config.json \
    --input targets/ --output batch_results/
```

### Use Case 5: Config Generator (`use_case_5_config.json`)

**Purpose**: Generate configurations from PDB files

**Key Settings**:
- `auto_suggest_hotspots: true` - Analyze structure for hotspots
- `validate_config: true` - Validate generated configs
- `copy_defaults: true` - Copy default filter files

**Typical Usage**:
```bash
python clean_scripts/use_case_5_config_generator.py \
    --config configs/use_case_5_config.json \
    --input target.pdb --output config_files/
```

## Creating Custom Configurations

### Template for New Config
```json
{
  "_description": "Custom configuration for specific use case",
  "_source": "Customized from use_case_X_config.json",

  "name": "CustomBinder",
  "chains": "A,B",
  "hotspot": "56,68,80",
  "binder_length": 150,
  "num_designs": 5,
  "device": "1",
  "filters_enabled": true,

  "processing": {
    "timeout": 7200,
    "asynchronous": true
  },

  "output": {
    "save_statistics": true,
    "include_metadata": true,
    "custom_parameter": "custom_value"
  }
}
```

### Validation

Use the config generator to validate configurations:
```bash
python clean_scripts/use_case_5_config_generator.py --input target.pdb --output test_config --validate
```

## Common Patterns

### High-Throughput Design
```json
{
  "num_designs": 10,
  "processing": {"asynchronous": true},
  "batch": {"max_concurrent": 2}
}
```

### Quick Testing
```json
{
  "num_designs": 1,
  "binder_length": 100,
  "processing": {"timeout": 1800}
}
```

### Multi-Chain Targets
```json
{
  "chains": "A,B,C",
  "hotspot": "56,125,200",
  "num_designs": 3
}
```

### GPU Optimization
```json
{
  "device": "0",
  "processing": {
    "batch_size": 32,
    "use_mixed_precision": true
  }
}
```

## Environment Variables

Some settings can be controlled via environment variables:

```bash
# Override default GPU
export CUDA_VISIBLE_DEVICES=1

# Set default timeout
export BINDCRAFT_TIMEOUT=7200

# Default output directory
export BINDCRAFT_OUTPUT_DIR=/path/to/results
```

## Troubleshooting Configurations

### Invalid JSON
```
JSONDecodeError: Expecting ',' delimiter
```
**Solution**: Check JSON syntax with a validator

### Unknown Parameters
```
Warning: Unknown parameter 'invalid_param' ignored
```
**Solution**: Check parameter names against documentation

### Path Issues
```
FileNotFoundError: Config file not found
```
**Solution**: Use absolute paths or check current directory

### Parameter Conflicts
```
Warning: synchronous=true but asynchronous=true, using synchronous
```
**Solution**: Remove conflicting parameters

## Best Practices

1. **Use descriptive names**: `"PDL1_high_affinity"` not `"test1"`
2. **Document custom configs**: Add `_description` field
3. **Validate configurations**: Use `--validate` flag
4. **Version control**: Keep configs in version control
5. **Environment-specific configs**: Different configs for different clusters/GPUs
6. **Backup configurations**: Save working configurations

## Examples for MCP Integration

The configurations are designed to work seamlessly with MCP tools:

```python
# MCP tool with default config
@mcp.tool()
def quick_design(input_file: str, output_file: str = None):
    config = load_json("configs/use_case_1_config.json")
    return run_quick_design(input_file, output_file, config=config)

# MCP tool with parameter override
@mcp.tool()
def custom_design(input_file: str, num_designs: int = 3, device: str = "0"):
    config = load_json("configs/use_case_1_config.json")
    config.update({"num_designs": num_designs, "device": device})
    return run_quick_design(input_file, config=config)
```