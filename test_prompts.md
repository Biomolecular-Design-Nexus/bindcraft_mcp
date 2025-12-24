# Test Prompts for MCP Integration

Use these prompts to test the MCP server integration with Claude Code:

## Basic Tool Discovery

### Prompt 1: List Available Tools
```
What MCP tools are available from the scripts server? List them with brief descriptions.
```

**Expected Response**: Should list all 12 tools with descriptions

### Prompt 2: Show Example Data
```
What example data files are available for testing the bindcraft tools?
```

**Expected Response**: Should show PDL1.pdb and config files

## Quick Sync Operations

### Prompt 3: Generate Configuration
```
Use the generate_config tool to analyze the PDL1.pdb file in examples/data/ and create a basic configuration for binder design.
```

**Expected Response**: Should generate configuration files and identify hotspots

### Prompt 4: Monitor Progress
```
Check if there are any jobs currently running using the monitor_progress tool.
```

**Expected Response**: Should show current job status or indicate no jobs running

## Job Management Workflow

### Prompt 5: Submit Async Job
```
Submit a binder design job using submit_async_design for the PDL1.pdb example file with 2 designs. Name the job "test_design".
```

**Expected Response**: Should return job ID and submission confirmation

### Prompt 6: Check Job Status
```
Check the status of the job I just submitted using get_job_status.
```

**Expected Response**: Should show job status, timestamps, and current state

### Prompt 7: View Job Logs
```
Show me the last 20 lines of logs for the running job using get_job_log.
```

**Expected Response**: Should display recent log entries from job execution

### Prompt 8: List All Jobs
```
List all submitted jobs and their current status using list_jobs.
```

**Expected Response**: Should show table of all jobs with status information

## Batch Processing

### Prompt 9: Batch Submission
```
Submit a batch design job for multiple files (you can use the examples directory as input) with submit_batch_design.
```

**Expected Response**: Should submit batch job and return batch job ID

## Error Handling Tests

### Prompt 10: Invalid File
```
Try to generate a config for a non-existent file "/fake/protein.pdb" to test error handling.
```

**Expected Response**: Should return structured error message about file not found

### Prompt 11: Invalid Job ID
```
Check the status of job "invalid123" to test error handling.
```

**Expected Response**: Should return error message about job not found

## End-to-End Workflow

### Prompt 12: Complete Design Pipeline
```
I want to design protein binders for PDL1. First analyze the structure, then submit a design job with 1 design, monitor its progress, and show me the results when complete.
```

**Expected Response**: Should execute full workflow:
1. Generate config for PDL1.pdb
2. Submit async design job
3. Monitor job status
4. Show logs and results

### Prompt 13: Troubleshooting Help
```
One of my jobs seems stuck. How can I check what's wrong and potentially cancel it?
```

**Expected Response**: Should suggest using get_job_log, get_job_status, and cancel_job tools

## Advanced Usage

### Prompt 14: Custom Parameters
```
Submit an async design job for PDL1.pdb with these specific parameters:
- 3 designs
- Target chain A
- Binder length 150
- Job name "custom_design"
```

**Expected Response**: Should submit job with custom parameters

### Prompt 15: Resource Management
```
List all currently running jobs and cancel any that have been running for more than expected time.
```

**Expected Response**: Should list jobs, identify long-running ones, and offer to cancel them

## Performance Testing

### Prompt 16: Rapid Tool Calls
```
Quickly execute these operations in sequence:
1. List example data
2. Generate config for PDL1.pdb
3. Check current jobs
4. Get available configs
```

**Expected Response**: Should complete all operations within seconds

## Integration Verification

### Prompt 17: Server Health Check
```
Verify that the MCP server is working properly by testing several different tool types.
```

**Expected Response**: Should test multiple tools and report overall health

### Prompt 18: Help and Documentation
```
I'm new to bindcraft. Can you explain what each tool does and give me a recommended workflow for designing protein binders?
```

**Expected Response**: Should explain tools and suggest step-by-step workflow

---

## Success Criteria

Each test prompt should:
- ✅ Execute without errors
- ✅ Return structured, informative responses
- ✅ Complete within reasonable time (sync tools <1s, job submission <1s)
- ✅ Provide clear next steps when applicable
- ✅ Handle errors gracefully with helpful messages

## Notes for Testing

1. **Tool Discovery**: Verify all 12 tools are available
2. **Sync Operations**: Should complete quickly (<1 second)
3. **Job Management**: Full lifecycle should work (submit → status → log → result/cancel)
4. **Error Handling**: Invalid inputs should return structured error messages
5. **Real-world Workflows**: Combined operations should work seamlessly

Use these prompts systematically to validate the MCP integration is working correctly in your Claude Code environment.