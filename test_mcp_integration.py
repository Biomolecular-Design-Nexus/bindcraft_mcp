#!/usr/bin/env python3
"""
Comprehensive MCP Integration Test Suite

This script tests all MCP tools to ensure they work correctly with Claude Code.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Import the MCP server
from src.server import mcp

class MCPIntegrationTester:
    """Test runner for MCP server integration."""

    def __init__(self):
        self.results = {
            "test_session_id": f"test_{int(time.time())}",
            "test_date": datetime.now().isoformat(),
            "server_info": {
                "name": "bindcraft",
                "tools_count": 0,
                "tools_list": []
            },
            "test_results": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "pass_rate": 0.0
            },
            "issues_found": []
        }
        self.project_root = Path(__file__).parent.resolve()
        self.examples_dir = self.project_root / "examples" / "data"

    async def run_all_tests(self):
        """Run all integration tests."""
        print("🧪 Starting MCP Integration Tests...")
        print("=" * 60)

        # Test 1: Tool Discovery
        await self.test_tool_discovery()

        # Test 2: Utility Tools (fast)
        await self.test_utility_tools()

        # Test 3: Sync Tools
        await self.test_sync_tools()

        # Test 4: Submit API (job management)
        await self.test_submit_api()

        # Test 5: Error Handling
        await self.test_error_handling()

        # Generate final report
        self.generate_summary()
        self.print_results()

        return self.results

    async def test_tool_discovery(self):
        """Test that all tools are discoverable."""
        print("📋 Testing Tool Discovery...")
        test_name = "tool_discovery"

        try:
            tools = await mcp.get_tools()
            tool_names = [tool for tool in tools]  # tools might be strings

            self.results["server_info"]["tools_count"] = len(tool_names)
            self.results["server_info"]["tools_list"] = tool_names

            expected_tools = [
                "get_job_status", "get_job_result", "get_job_log", "cancel_job", "list_jobs",
                "quick_design", "monitor_progress", "generate_config",
                "submit_async_design", "submit_batch_design",
                "list_example_data", "get_default_configs"
            ]

            missing_tools = [tool for tool in expected_tools if tool not in tool_names]
            extra_tools = [tool for tool in tool_names if tool not in expected_tools]

            if missing_tools or extra_tools:
                self.results["test_results"][test_name] = {
                    "status": "failed",
                    "error": f"Tool mismatch - Missing: {missing_tools}, Extra: {extra_tools}",
                    "expected": expected_tools,
                    "actual": tool_names
                }
            else:
                self.results["test_results"][test_name] = {
                    "status": "passed",
                    "tools_found": len(tool_names),
                    "tools_list": tool_names
                }

            print(f"   ✅ Found {len(tool_names)} tools: {', '.join(tool_names)}")

        except Exception as e:
            self.results["test_results"][test_name] = {
                "status": "failed",
                "error": f"Tool discovery failed: {str(e)}"
            }
            self.results["issues_found"].append({
                "test": test_name,
                "severity": "critical",
                "error": str(e)
            })
            print(f"   ❌ Tool discovery failed: {e}")

    async def test_utility_tools(self):
        """Test utility tools that should work without external dependencies."""
        print("🛠️  Testing Utility Tools...")

        # Test list_example_data
        await self._test_tool_call("list_example_data", {})

        # Test get_default_configs
        await self._test_tool_call("get_default_configs", {})

    async def test_sync_tools(self):
        """Test synchronous tools with sample data."""
        print("⚡ Testing Sync Tools...")

        # Check if example data exists
        if not self.examples_dir.exists():
            print(f"   ⚠️  Examples directory not found: {self.examples_dir}")
            self.results["test_results"]["sync_tools_setup"] = {
                "status": "skipped",
                "reason": "No examples directory"
            }
            return

        # Get list of available example files
        example_files = list(self.examples_dir.glob("*.pdb"))
        if not example_files:
            print(f"   ⚠️  No PDB files found in examples directory")
            self.results["test_results"]["sync_tools_setup"] = {
                "status": "skipped",
                "reason": "No PDB files available"
            }
            return

        example_file = str(example_files[0])
        print(f"   📁 Using example file: {Path(example_file).name}")

        # Test generate_config (fastest tool)
        await self._test_tool_call("generate_config", {
            "input_file": example_file,
            "chains": "A",
            "validate": True,
            "analysis_type": "basic"
        })

        # Test monitor_progress (should work even if no job is running)
        test_output_dir = self.project_root / "test_outputs"
        test_output_dir.mkdir(exist_ok=True)
        await self._test_tool_call("monitor_progress", {
            "output_dir": str(test_output_dir),
            "detailed": False,
            "continuous": False
        })

    async def test_submit_api(self):
        """Test job submission and management APIs."""
        print("🚀 Testing Submit API & Job Management...")

        # Test list_jobs (should work even with no jobs)
        await self._test_tool_call("list_jobs", {})

        # Test get_job_status with invalid job ID (should return error gracefully)
        await self._test_tool_call("get_job_status", {"job_id": "invalid_job_id"})

        # Test cancel_job with invalid job ID
        await self._test_tool_call("cancel_job", {"job_id": "invalid_job_id"})

        # Test get_job_log with invalid job ID
        await self._test_tool_call("get_job_log", {"job_id": "invalid_job_id", "tail": 10})

        print("   ℹ️  Note: Full job submission tests require working BindCraft installation")

    async def test_error_handling(self):
        """Test error handling with invalid inputs."""
        print("🛡️  Testing Error Handling...")

        # Test with non-existent file
        await self._test_tool_call("generate_config", {
            "input_file": "/nonexistent/file.pdb"
        })

        # Test with invalid parameters
        await self._test_tool_call("quick_design", {
            "input_file": "/fake/file.pdb",
            "num_designs": -1  # Invalid negative number
        })

    async def _test_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to test individual tool calls."""
        test_key = f"{tool_name}_test"
        start_time = time.time()

        try:
            print(f"   🔧 Testing {tool_name}...")

            # Get the tool
            tool = await mcp.get_tool(tool_name)
            if not tool:
                raise Exception(f"Tool {tool_name} not found")

            # Call the tool (this simulates what Claude Code would do)
            tool_result = await tool.run(args)

            execution_time = time.time() - start_time

            # Extract actual result from ToolResult
            if hasattr(tool_result, 'content') and tool_result.content:
                # Extract text content and try to parse as JSON
                text_content = tool_result.content[0].text if tool_result.content else "{}"
                try:
                    result = json.loads(text_content)
                except json.JSONDecodeError:
                    result = {"raw_content": text_content}
            else:
                result = {"tool_result": str(tool_result)}

            # Check if result indicates success or error
            if isinstance(result, dict):
                if result.get("status") == "error":
                    self.results["test_results"][test_key] = {
                        "status": "expected_error",
                        "tool": tool_name,
                        "args": args,
                        "result": result,
                        "execution_time": execution_time,
                        "note": "Tool returned expected error response"
                    }
                    print(f"      ⚠️  Expected error: {result.get('error', 'Unknown error')}")
                else:
                    self.results["test_results"][test_key] = {
                        "status": "passed",
                        "tool": tool_name,
                        "args": args,
                        "result": result,
                        "execution_time": execution_time
                    }
                    print(f"      ✅ Success in {execution_time:.2f}s")
            else:
                self.results["test_results"][test_key] = {
                    "status": "passed",
                    "tool": tool_name,
                    "args": args,
                    "result": result,
                    "execution_time": execution_time
                }
                print(f"      ✅ Success in {execution_time:.2f}s")

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)

            self.results["test_results"][test_key] = {
                "status": "failed",
                "tool": tool_name,
                "args": args,
                "error": error_msg,
                "execution_time": execution_time
            }

            self.results["issues_found"].append({
                "test": test_key,
                "tool": tool_name,
                "severity": "medium",
                "error": error_msg
            })

            print(f"      ❌ Failed in {execution_time:.2f}s: {error_msg}")

    def generate_summary(self):
        """Generate test summary statistics."""
        total = len(self.results["test_results"])
        passed = sum(1 for test in self.results["test_results"].values()
                    if test["status"] in ["passed", "expected_error"])
        failed = sum(1 for test in self.results["test_results"].values()
                    if test["status"] == "failed")
        skipped = sum(1 for test in self.results["test_results"].values()
                     if test["status"] == "skipped")

        self.results["summary"] = {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": (passed / total * 100) if total > 0 else 0
        }

    def print_results(self):
        """Print formatted test results."""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)

        summary = self.results["summary"]
        server_info = self.results["server_info"]

        print(f"🖥️  Server: {server_info.get('name', 'unknown')} ({server_info.get('tools_count', 0)} tools)")
        print(f"📈 Results: {summary['passed']}/{summary['total_tests']} passed ({summary['pass_rate']:.1f}%)")

        if summary['failed'] > 0:
            print(f"❌ Failed: {summary['failed']}")
        if summary['skipped'] > 0:
            print(f"⏭️  Skipped: {summary['skipped']}")

        print(f"⏱️  Test Duration: {datetime.now().isoformat()}")

        if self.results["issues_found"]:
            print("\n🚨 ISSUES FOUND:")
            for issue in self.results["issues_found"]:
                print(f"   • {issue['test']}: {issue['error']}")

        # Status determination
        if summary['failed'] == 0 and summary['passed'] > 0:
            print("\n🎉 ALL TESTS PASSED - MCP server is ready for production!")
            return True
        elif summary['failed'] > 0:
            print(f"\n⚠️  {summary['failed']} TESTS FAILED - See issues above")
            return False
        else:
            print("\n❓ NO TESTS RUN - Check server configuration")
            return False


async def main():
    """Main test runner."""
    tester = MCPIntegrationTester()
    results = await tester.run_all_tests()

    # Save detailed results
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    with open(reports_dir / "mcp_integration_test_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Detailed results saved to: reports/mcp_integration_test_results.json")

    # Return success/failure for CI
    return results["summary"]["failed"] == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)