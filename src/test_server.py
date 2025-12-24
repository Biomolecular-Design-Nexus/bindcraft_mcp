#!/usr/bin/env python3
"""Simple test script to verify MCP server functionality."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_job_management():
    """Test job management tools."""
    from jobs.manager import job_manager

    print("Testing Job Management...")

    # Test list_jobs
    result = job_manager.list_jobs()
    print(f"✓ list_jobs: {result['status']}, found {result['total']} jobs")

    print("Job management tests passed!")

def test_server_import():
    """Test that server imports correctly."""
    print("Testing server import...")

    try:
        from src.server import mcp
        print("✓ Server imported successfully")

        # Get tool list
        tools = mcp._tools
        tool_names = list(tools.keys())
        print(f"✓ Found {len(tool_names)} tools: {', '.join(tool_names)}")

        return True
    except Exception as e:
        print(f"✗ Server import failed: {e}")
        return False

def test_utility_tools():
    """Test utility tools that don't require external dependencies."""
    print("Testing utility tools...")

    try:
        from src.server import mcp

        # Test list_example_data
        result = mcp._tools["list_example_data"]._func()
        print(f"✓ list_example_data: {result['status']}")

        # Test get_default_configs
        result = mcp._tools["get_default_configs"]._func()
        print(f"✓ get_default_configs: {result['status']}")
        if result["status"] == "success":
            print(f"  Found {result['total_configs']} config files")

        return True
    except Exception as e:
        print(f"✗ Utility tools failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing BindCraft MCP Server ===\n")

    success = True

    # Test imports
    success &= test_server_import()
    print()

    # Test job management
    try:
        test_job_management()
        print()
    except Exception as e:
        print(f"✗ Job management tests failed: {e}\n")
        success = False

    # Test utility tools
    success &= test_utility_tools()
    print()

    if success:
        print("🎉 All tests passed! MCP server is ready.")
    else:
        print("❌ Some tests failed. Check the errors above.")
        sys.exit(1)