#!/usr/bin/env python3
"""Test script to list available MCP tools."""

import asyncio
from src.bindcraft_mcp import mcp

async def test_tools():
    try:
        tools = await mcp.get_tools()
        print(f'Found {len(tools)} tools:')
        for i, tool in enumerate(tools):
            if hasattr(tool, 'name'):
                name = tool.name
                desc = getattr(tool, 'description', 'No description')
            elif isinstance(tool, str):
                name = tool
                desc = 'String tool name'
            else:
                name = f"Tool {i}"
                desc = f"Type: {type(tool)}"
            print(f'  - {name}: {desc[:80]}...' if len(str(desc)) > 80 else f'  - {name}: {desc}')
        return True
    except Exception as e:
        print(f"Error listing tools: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_tools())
    exit(0 if success else 1)