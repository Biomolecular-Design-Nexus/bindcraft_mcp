"""
Model Context Protocol (MCP) for BindCraft

This MCP server provides protein binder design tools using BindCraft.
It enables researchers to design de novo protein binders against target proteins
using AlphaFold2 hallucination and ProteinMPNN sequence optimization.

This MCP Server contains tools for:

1. bindcraft_design_binder
   - Design de novo protein binders against a target protein structure (synchronous)
   - Uses AF2 for structure hallucination and MPNN for sequence optimization
   - Supports multiple design algorithms (2stage, 3stage, 4stage, greedy, mcmc)
   - Waits for completion and returns full results

2. bindcraft_submit
   - Submit a BindCraft job asynchronously (returns immediately)
   - Supports both config file input and simple PDB parameters
   - Returns 'submitted' status with output directory for monitoring
   - Job runs in background independently

3. bindcraft_check_status
   - Check status and results of a submitted or running job
   - Returns statistics on accepted/rejected designs and trajectories

4. generate_config
   - Generate BindCraft configuration files from PDB structures
   - Analyzes PDB and creates target_settings.json for design jobs
   - Fast, no GPU required

5. validate_config
   - Validate configuration files before job submission
   - Checks required fields, types, and file existence
   - Helps catch errors early

Workflow Overview:
1. AF2 Hallucination: Generate binder backbone conformations
2. MPNN Sequence Design: Optimize sequences for the binder backbone
3. AF2 Prediction: Validate designed binders with structure prediction
4. PyRosetta Scoring: Evaluate interface quality and energy

Usage:
    # Run the MCP server
    python bindcraft_mcp.py

    # Or use with uvicorn for production
    uvicorn bindcraft_mcp:mcp --host 0.0.0.0 --port 8000
"""

from loguru import logger
from fastmcp import FastMCP

# Import tool MCPs
from tools.bindcraft_design import bindcraft_design_mcp
from tools.bindcraft_config import bindcraft_config_mcp

# Server definition and mounting
mcp = FastMCP(name="bindcraft")
logger.info("Mounting bindcraft_design tool")
mcp.mount(bindcraft_design_mcp)
logger.info("Mounting bindcraft_config tool")
mcp.mount(bindcraft_config_mcp)

if __name__ == "__main__":
    logger.info("Starting BindCraft MCP server")
    mcp.run()
