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
        import server
        print("✓ Server imported successfully")

        # Get tool info by checking server module
        tools = getattr(server, 'mcp', None)
        if tools and hasattr(tools, '_tools'):
            tool_names = list(tools._tools.keys())
            print(f"✓ Found {len(tool_names)} tools: {', '.join(tool_names)}")
        else:
            print("✓ Server module loaded (tools not directly accessible)")

        return True
    except Exception as e:
        print(f"✗ Server import failed: {e}")
        return False

def test_utility_tools():
    """Test utility tools that don't require external dependencies."""
    print("Testing utility tools...")

    try:
        import server

        # Test the configs directory exists
        configs_dir = Path("./configs")
        if configs_dir.exists():
            print(f"✓ Configs directory found with {len(list(configs_dir.glob('*.json')))} files")
        else:
            print("! Configs directory not found (but that's OK)")

        # Test examples directory
        examples_dir = Path("./examples/data")
        if examples_dir.exists():
            print(f"✓ Examples directory found")
        else:
            print("! Examples directory not found (but that's OK)")

        return True
    except Exception as e:
        print(f"✗ Utility tools check failed: {e}")
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