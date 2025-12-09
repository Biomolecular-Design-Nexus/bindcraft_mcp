# BindCraft MCP - Usage Examples

## Overview

The BindCraft MCP provides three main tools for protein binder design:

1. **bindcraft_submit** - Submit jobs asynchronously (returns immediately)
2. **bindcraft_check_status** - Check job status and get results
3. **bindcraft_design_binder** - Run synchronously (waits for completion)

## Example 1: Submit Job with PDB File (Simple Mode)

```python
# Submit a binder design job
result = bindcraft_submit(
    target_pdb="/path/to/target_protein.pdb",
    output_dir="/path/to/output",
    binder_name="MyBinder",
    target_chains="A",
    hotspot_residues="56,78,90",
    min_binder_length=65,
    max_binder_length=150,
    num_designs=10,
    device=0
)

# Returns immediately with:
# {
#     "status": "submitted",
#     "message": "BindCraft job submitted successfully...",
#     "output_dir": "/path/to/output",
#     "log_file": "/path/to/output/bindcraft_run.log",
#     "pid": 12345,
#     "target_settings": {...}
# }

print(f"Job submitted! Output directory: {result['output_dir']}")
print(f"Process ID: {result['pid']}")
```

## Example 2: Submit Job with Config File

```python
# Submit using a pre-made configuration file
result = bindcraft_submit(
    settings_json="/path/to/target_settings.json",
    device=0
)

# Job runs in background
output_dir = result['output_dir']
```

## Example 3: Monitor Job Progress

```python
# Check status while job is running
status = bindcraft_check_status(output_dir=output_dir)

print(f"Job status: {status['job_status']}")
# Possible values:
# - "not_started": Log file doesn't exist yet
# - "running": Job is actively running (log updated recently)
# - "possibly_running": Log hasn't updated in a while
# - "completed": Job finished successfully
# - "failed": Job encountered errors
# - "stalled_or_completed": Log is old, unclear status

# View current progress
print(f"Accepted designs: {status['statistics']['accepted_designs']}")
print(f"Rejected designs: {status['statistics']['rejected_designs']}")
print(f"Trajectories: {status['total_trajectories']}")
```

## Example 4: Get Final Results (Job Completed)

```python
# When job is finished, get comprehensive summary
status = bindcraft_check_status(output_dir=output_dir)

if status['job_status'] in ['completed', 'failed']:
    summary = status['summary']

    print(f"Status: {summary['completion_status']}")
    print(f"Message: {summary['message']}")

    # Results breakdown
    results = summary['results']
    print(f"Accepted: {results['accepted_designs']}")
    print(f"Rejected: {results['rejected_designs']}")
    print(f"Total evaluated: {results['total_evaluated']}")
    print(f"Trajectories generated: {results['trajectories_generated']}")

    # Success metrics
    if 'success_rate' in summary:
        print(f"Success rate: {summary['success_rate']}")
    if 'acceptance_rate' in summary:
        print(f"Acceptance rate: {summary['acceptance_rate']}")

    # Target information
    target = summary['target']
    print(f"\nTarget: {target['binder_name']}")
    print(f"PDB: {target['target_pdb']}")
    print(f"Chains: {target['chains']}")

    # View log tail
    if 'log_tail' in summary:
        print("\nRecent log entries:")
        for line in summary['log_tail'][-5:]:
            print(f"  {line}")

    # Check for errors (if failed)
    if 'recent_errors' in summary:
        print("\nRecent errors:")
        for error in summary['recent_errors']:
            print(f"  {error}")
```

## Example 5: Complete Workflow

```python
# 1. Submit job
print("Submitting BindCraft job...")
result = bindcraft_submit(
    target_pdb="examples/PDL1.pdb",
    output_dir="output/PDL1_binders",
    binder_name="PDL1",
    target_chains="A",
    num_designs=20,
    device=0
)

output_dir = result['output_dir']
print(f"✓ Job submitted to {output_dir}")

# 2. Monitor progress (you can check periodically)
import time

while True:
    status = bindcraft_check_status(output_dir=output_dir)
    job_status = status['job_status']

    print(f"Status: {job_status}")
    print(f"  Accepted: {status['statistics']['accepted_designs']}")
    print(f"  Rejected: {status['statistics']['rejected_designs']}")

    if job_status in ['completed', 'failed']:
        break

    time.sleep(60)  # Check every minute

# 3. Get final results
if job_status == 'completed':
    summary = status['summary']
    print(f"\n✓ Design completed!")
    print(f"  {summary['message']}")
    print(f"  Success rate: {summary.get('success_rate', 'N/A')}")

    # List accepted designs
    print(f"\nAccepted designs (in {output_dir}/Accepted):")
    for design_file in status['accepted_designs'][:10]:
        print(f"  - {design_file}")
else:
    print(f"\n✗ Job failed")
    print(f"  Check log file: {status['log_file']}")
```

## Job Status Values

| Status | Description |
|--------|-------------|
| `not_started` | Log file doesn't exist yet |
| `running` | Job is active (log updated < 5 min ago) |
| `possibly_running` | Log updated within last hour |
| `stalled_or_completed` | Log is old (> 1 hour) |
| `completed` | Job finished successfully |
| `failed` | Job encountered errors |
| `unknown` | Cannot determine status |

## Summary Fields (When Job is Finished)

When `job_status` is `completed` or `failed`, the status response includes a `summary` field:

```python
{
    "summary": {
        "job_status": "completed",
        "completion_status": "Success",  # or "Failed"
        "message": "Design completed successfully with 8 accepted binder(s).",

        "target": {
            "binder_name": "MyBinder",
            "target_pdb": "/path/to/target.pdb",
            "chains": "A",
            "hotspot_residues": "56,78,90",
            "binder_length_range": [65, 150],
            "requested_designs": 10
        },

        "results": {
            "accepted_designs": 8,
            "rejected_designs": 12,
            "total_evaluated": 20,
            "trajectories_generated": 25,
            "trajectories_relaxed": 20,
            "trajectories_clashing": 3,
            "trajectories_low_confidence": 2,
            "mpnn_designs": 15
        },

        "success_rate": "80.0% of target (8/10)",
        "acceptance_rate": "40.0% (8/20)",

        "log_tail": ["...", "..."],  # Last 20 log lines
        "recent_errors": ["..."]      # Only if job failed
    }
}
```

## Tips

1. **Use `bindcraft_submit` for long-running jobs** - It returns immediately and you can monitor progress separately.

2. **Monitor with `bindcraft_check_status`** - Check periodically to see progress and get final results.

3. **Check the log file** - The log file path is provided in the submission response for detailed debugging.

4. **Use synchronous mode for testing** - Use `bindcraft_design_binder` when you want to wait for completion (useful for debugging).

5. **Output directory structure**:
   ```
   output_dir/
   ├── bindcraft_run.log           # Job log file
   ├── target_settings.json        # Job configuration
   ├── Accepted/                   # Successful binder designs
   ├── Rejected/                   # Rejected designs
   ├── Trajectory/                 # Intermediate structures
   │   ├── Relaxed/
   │   ├── Clashing/
   │   └── LowConfidence/
   └── MPNN/                       # MPNN outputs
   ```
