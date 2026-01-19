# BindCraft MCP

> Model Context Protocol (MCP) server for protein binder design using BindCraft

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

The BindCraft MCP provides powerful tools for protein binder design through both synchronous and asynchronous APIs. It enables computational design of proteins that bind to specific target proteins, leveraging advanced machine learning models including AlphaFold2, MPNN, and PyRosetta for structure prediction, sequence design, and optimization.

### Features
- **Quick Design**: Fast synchronous binder design for single targets (1-10 minutes)
- **Async Processing**: Long-running background jobs for complex designs (>10 minutes)
- **Batch Processing**: Multi-target processing with concurrent job management
- **Job Management**: Complete lifecycle tracking with status, logs, and result retrieval
- **Config Generation**: Automatic parameter optimization from PDB structures
- **Progress Monitoring**: Real-time tracking of running design jobs
- **GPU Acceleration**: Full CUDA support with JAX/XLA optimization
- **Error Handling**: Robust error reporting and graceful degradation

### Directory Structure
```
./
├── README.md               # This file
├── env/                    # Conda environment
├── src/
│   ├── bindcraft_mcp.py    # MCP server (main entry point)
│   └── tools/
│       ├── bindcraft_design.py   # Design tools
│       └── bindcraft_config.py   # Config tools
├── clean_scripts/
│   ├── use_case_1_quick_design.py        # Quick synchronous design
│   ├── use_case_2_async_submission.py    # Async job submission
│   ├── use_case_3_monitor_progress.py    # Job monitoring
│   ├── use_case_4_batch_design.py        # Batch processing
│   ├── use_case_5_config_generator.py    # Config generation
│   └── lib/                # Shared utilities
├── examples/
│   └── data/               # Demo data (PDL1.pdb, configs)
├── configs/                # Configuration files
└── repo/                   # Original BindCraft repository
```

---

## Installation

### Quick Setup (Recommended)

Run the automated setup script:

```bash
cd bindcraft_mcp
bash quick_setup.sh
```

The script will create the conda environment, clone the BindCraft repository, install all dependencies, and display the Claude Code configuration. See `quick_setup.sh --help` for options like `--skip-env` or `--skip-repo`.

### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+
- NVIDIA GPU with CUDA support (recommended for design tasks)

### Manual Installation (Alternative)

If you prefer manual installation or need to customize the setup, follow `reports/step3_environment.md`:

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/bindcraft_mcp

# Create conda environment (use mamba if available)
mamba create -p ./env python=3.10 -y
# or: conda create -p ./env python=3.10 -y

# Activate environment
mamba activate ./env
# or: conda activate ./env

# Install Dependencies
# Core scientific stack
mamba install -c conda-forge numpy pandas biopython click -y

# Install JAX with CUDA support
pip install jax[cuda12] jaxlib

# Install MCP dependencies
pip install fastmcp loguru
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Example |
|--------|-------------|---------|
| `clean_scripts/use_case_1_quick_design.py` | Quick synchronous binder design | See below |
| `clean_scripts/use_case_2_async_submission.py` | Async job submission for background processing | See below |
| `clean_scripts/use_case_3_monitor_progress.py` | Monitor running BindCraft jobs | See below |
| `clean_scripts/use_case_4_batch_design.py` | Batch processing for multiple targets | See below |
| `clean_scripts/use_case_5_config_generator.py` | Generate configuration files from PDB | See below |

### Script Examples

#### Quick Design (Synchronous)

```bash
# Activate environment
mamba activate ./env

# Run quick design
python clean_scripts/use_case_1_quick_design.py \
  --input examples/data/PDL1.pdb \
  --output results/quick_design \
  --num-designs 1 \
  --chains A \
  --device 0
```

**Parameters:**
- `--input, -i`: Path to input PDB file (required)
- `--output, -o`: Output directory for results (default: auto-generated)
- `--num-designs`: Number of designs to generate (default: 1)
- `--chains`: Target chains (default: "A")
- `--binder-length`: Binder sequence length (default: 130)
- `--device`: GPU device number (default: 0)
- `--hotspot`: Specific hotspot residues (optional)
- `--config`: Custom config file path (optional)

#### Async Job Submission

```bash
python clean_scripts/use_case_2_async_submission.py \
  --input examples/data/PDL1.pdb \
  --output results/async_design \
  --num-designs 3 \
  --device 0
```

**Parameters:**
- `--input, -i`: Path to input PDB file (required)
- `--output, -o`: Output directory for results (default: auto-generated)
- `--num-designs`: Number of designs to generate (default: 3)
- `--chains`: Target chains (default: "A")
- `--binder-length`: Binder sequence length (default: 130)
- `--device`: GPU device number (default: 0)
- `--config`: Custom config file path (optional)

#### Monitor Progress

```bash
python clean_scripts/use_case_3_monitor_progress.py \
  --output results/async_design \
  --detailed \
  --interval 30
```

**Parameters:**
- `--output, -o`: Directory containing job to monitor (required)
- `--detailed`: Include detailed progress information (flag)
- `--continuous`: Enable continuous monitoring (flag)
- `--interval`: Monitoring interval in seconds (default: 30)

#### Batch Processing

```bash
python clean_scripts/use_case_4_batch_design.py \
  --input examples/data/ \
  --output results/batch_design \
  --num-designs 1 \
  --max-concurrent 2
```

**Parameters:**
- `--input, -i`: Path to directory with PDB files or single PDB (required)
- `--output, -o`: Base output directory (default: auto-generated)
- `--num-designs`: Number of designs per target (default: 1)
- `--max-concurrent`: Maximum concurrent jobs (default: 3)
- `--chains`: Target chains (default: "A")
- `--device`: GPU device number (default: 0)

#### Configuration Generation

```bash
python clean_scripts/use_case_5_config_generator.py \
  --input examples/data/PDL1.pdb \
  --output results/config_gen \
  --validate \
  --chains A
```

**Parameters:**
- `--input, -i`: Path to input PDB file (required)
- `--output, -o`: Output directory for configs (default: auto-generated)
- `--chains`: Chains to analyze (default: "A")
- `--binder-length`: Target binder length (default: 130)
- `--validate`: Validate generated config (flag)
- `--analysis-type`: Analysis type (basic, detailed, advanced)

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Install MCP server for Claude Code
fastmcp install src/bindcraft_mcp.py --name bindcraft
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add bindcraft -- $(pwd)/env/bin/python $(pwd)/src/bindcraft_mcp.py

# Verify installation
claude mcp list
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "bindcraft": {
      "command": "/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/bindcraft_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/bindcraft_mcp/src/bindcraft_mcp.py"]
    }
  }
}
```

---

## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What tools are available from bindcraft?
```

#### Basic Usage
```
Use quick_design with input_file @examples/data/PDL1.pdb
```

#### With Configuration
```
Generate a binder for @examples/data/PDL1.pdb with 3 designs targeting chain A
```

#### Long-Running Tasks (Submit API)
```
Submit an async binder design job for @examples/data/PDL1.pdb with num_designs 3
Then check the job status
```

#### Batch Processing
```
Submit batch design for all PDB files in @examples/data/ directory
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/data/PDL1.pdb` | Reference the sample PDB file |
| `@configs/default_config.json` | Reference a config file |
| `@results/` | Reference output directory |

---

## Using with Gemini CLI

### Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "bindcraft": {
      "command": "/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/bindcraft_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/bindcraft_mcp/src/bindcraft_mcp.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same as Claude Code)
> What tools are available?
> Use quick_design with file examples/data/PDL1.pdb
> Submit async design for PDL1.pdb with 3 designs
```

---

## Available Tools

The BindCraft MCP provides 5 tools for protein binder design:

### Design Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `bindcraft_design_binder` | Synchronous binder design | `pdb_path`, `output_dir`, `target_chains`, `binder_length`, `num_designs`, `hotspot_residues`, `device` |
| `bindcraft_submit` | Submit async design job | `pdb_path`, `output_dir`, `target_chains`, `binder_length`, `num_designs`, `hotspot_residues`, `device` |
| `bindcraft_check_status` | Check async job status | `output_dir` |

### Configuration Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `generate_config` | Generate config from PDB | `pdb_path`, `output_path`, `target_chains`, `binder_length`, `analysis_type` |
| `validate_config` | Validate config file | `config_path`, `strict` |

---

## Examples

### Example 1: Quick Binder Design

**Goal:** Design a single binder for PDL1 protein quickly

**Using Script:**
```bash
python clean_scripts/use_case_1_quick_design.py \
  --input examples/data/PDL1.pdb \
  --output results/example1 \
  --num-designs 1 \
  --chains A
```

**Using MCP (in Claude Code):**
```
Use quick_design to process @examples/data/PDL1.pdb and save results to results/example1/
```

**Expected Output:**
- Binder PDB structures in `results/example1/accepted/`
- Design statistics and metadata
- Target settings configuration file

### Example 2: Async Job with Monitoring

**Goal:** Submit a multi-design job and monitor progress

**Using Script:**
```bash
# Submit job
python clean_scripts/use_case_2_async_submission.py \
  --input examples/data/PDL1.pdb \
  --output results/example2 \
  --num-designs 3

# Monitor progress
python clean_scripts/use_case_3_monitor_progress.py \
  --output results/example2 \
  --detailed \
  --continuous
```

**Using MCP (in Claude Code):**
```
Submit async binder design for @examples/data/PDL1.pdb with num_designs 3

Then check the job status and show me the logs
```

### Example 3: Configuration Generation and Validation

**Goal:** Generate optimized configuration from PDB structure

**Using Script:**
```bash
python clean_scripts/use_case_5_config_generator.py \
  --input examples/data/PDL1.pdb \
  --output results/example3 \
  --validate \
  --analysis-type detailed
```

**Using MCP (in Claude Code):**
```
Generate configuration for @examples/data/PDL1.pdb with detailed analysis and validation
```

**Expected Output:**
- `target_settings.json` with optimized parameters
- Structure analysis results with hotspot suggestions
- Validation report

### Example 4: Batch Processing

**Goal:** Process multiple targets simultaneously

**Using Script:**
```bash
# Create a directory with multiple PDB files
mkdir -p examples/batch_input
cp examples/data/PDL1.pdb examples/batch_input/target1.pdb
cp examples/data/PDL1.pdb examples/batch_input/target2.pdb

python clean_scripts/use_case_4_batch_design.py \
  --input examples/batch_input/ \
  --output results/example4 \
  --num-designs 1 \
  --max-concurrent 2
```

**Using MCP (in Claude Code):**
```
Submit batch processing for all PDB files in @examples/batch_input/ with max_concurrent 2
```

---

## Demo Data

The `examples/data/` directory contains sample data for testing:

| File | Description | Use With |
|------|-------------|----------|
| `PDL1.pdb` | PDL1 protein structure (74KB, 115 residues) | All design tools |
| `default_filters.json` | Filter settings for design pipeline (28KB) | Design scripts with filtering |
| `default_4stage_multimer.json` | Advanced 4-stage algorithm parameters (2KB) | Complex design workflows |

---

## Configuration Files

The `configs/` directory contains configuration templates:

| Config | Description | Parameters |
|--------|-------------|------------|
| `default_config.json` | Common defaults for all scripts | device, timeout, binder_length, etc. |
| `use_case_1_config.json` | Quick design settings | synchronous design parameters |
| `use_case_2_config.json` | Async submission settings | background execution parameters |
| `use_case_3_config.json` | Monitoring settings | refresh intervals, analysis settings |
| `use_case_4_config.json` | Batch processing config | concurrency, batch management |
| `use_case_5_config.json` | Config generation settings | analysis type, validation options |

### Config Example

```json
{
  "general": {
    "device": "0",
    "timeout": 3600
  },
  "design": {
    "name": "Binder",
    "chains": "A",
    "binder_length": 130,
    "num_designs": 1,
    "filters_enabled": true
  },
  "processing": {
    "synchronous": true,
    "background_execution": false,
    "log_file": "bindcraft_run.log"
  }
}
```

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Recreate environment
mamba create -p ./env python=3.10 -y
mamba activate ./env
pip install jax[cuda12] jaxlib fastmcp loguru biopython pandas numpy click
```

**Problem:** Import errors
```bash
# Verify installation
python -c "from src.bindcraft_mcp import mcp"
python -c "import jax; print(jax.devices())"  # Check GPU availability
```

**Problem:** CUDA not available
```bash
# Check CUDA installation
nvidia-smi
python -c "import jax; print(jax.devices())"
# If no GPU, CPU mode will be used (slower)
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove bindcraft
claude mcp add bindcraft -- $(pwd)/env/bin/python $(pwd)/src/bindcraft_mcp.py
```

**Problem:** Tools not working
```bash
# Test server directly
python src/bindcraft_mcp.py &
# Test with MCP inspector
npx @anthropic/mcp-inspector src/bindcraft_mcp.py
```

### Design Issues

**Problem:** Design job failed
- Check input file paths are absolute
- Verify BindCraft dependencies in `repo/scripts/`
- Check GPU availability with `nvidia-smi`
- Review output directory for log files

**Problem:** Check status returns not found
- Verify the output_dir path is correct
- Check if the design process has started
- Look for `.running` or `progress.json` files in output directory

### BindCraft Dependencies

**Problem:** BindCraft scripts not found
```bash
# Ensure repo directory exists
ls -la repo/scripts/run_bindcraft.py

# If missing, the original BindCraft repository is needed
# for use cases 1, 2, and 4 (design scripts)
```

**Problem:** Design fails with import errors
- Ensure the environment contains all BindCraft dependencies
- Check that JAX, BioPython, and PyRosetta are available
- Verify GPU drivers and CUDA installation

---

## Development

### Running Tests

```bash
# Activate environment
mamba activate ./env

# Test individual scripts
python clean_scripts/use_case_5_config_generator.py --input examples/data/PDL1.pdb --output test_config
python clean_scripts/use_case_3_monitor_progress.py --output nonexistent_dir

# Test MCP tools
python tests/test_tools.py
```

### Starting Dev Server

```bash
# Run MCP server in dev mode
fastmcp dev src/bindcraft_mcp.py

# Or test with inspector
npx @anthropic/mcp-inspector src/bindcraft_mcp.py
```

### Performance Monitoring

```bash
# Check GPU usage
nvidia-smi -l 1

# View design logs in output directories
tail -f results/*/bindcraft_run.log
```

---

## License

Based on the original BindCraft repository. Please refer to the original repository for licensing terms.

## Credits

Based on [BindCraft](https://github.com/martinpacesa/BindCraft) by Martin Pacesa and colleagues.

This MCP implementation provides a clean interface to BindCraft's powerful protein binder design capabilities, making them accessible through modern AI assistants like Claude Code.