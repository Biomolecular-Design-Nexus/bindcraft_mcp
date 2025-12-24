"""Job management for long-running tasks."""

import uuid
import json
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import sys
import os

# Add clean_scripts to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "clean_scripts"))

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobManager:
    """Manages asynchronous job execution."""

    def __init__(self, jobs_dir: Path = None):
        self.jobs_dir = jobs_dir or Path(__file__).parent.parent.parent / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._running_jobs: Dict[str, subprocess.Popen] = {}

    def submit_job(
        self,
        script_path: str,
        args: Dict[str, Any],
        job_name: str = None
    ) -> Dict[str, Any]:
        """Submit a new job for background execution.

        Args:
            script_path: Path to the script to run
            args: Arguments to pass to the script
            job_name: Optional name for the job

        Returns:
            Dict with job_id and status
        """
        job_id = str(uuid.uuid4())[:8]
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Save job metadata
        metadata = {
            "job_id": job_id,
            "job_name": job_name or f"job_{job_id}",
            "script": script_path,
            "args": args,
            "status": JobStatus.PENDING.value,
            "submitted_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None
        }

        self._save_metadata(job_id, metadata)

        # Start job in background
        self._start_job(job_id, script_path, args, job_dir)

        return {
            "status": "submitted",
            "job_id": job_id,
            "message": f"Job submitted. Use get_job_status('{job_id}') to check progress."
        }

    def _start_job(self, job_id: str, script_path: str, args: Dict, job_dir: Path):
        """Start job execution in background thread."""
        def run_job():
            metadata = self._load_metadata(job_id)
            metadata["status"] = JobStatus.RUNNING.value
            metadata["started_at"] = datetime.now().isoformat()
            self._save_metadata(job_id, metadata)

            try:
                # Build command - scripts are run from their parent directory
                script_full_path = Path(script_path)
                if not script_full_path.is_absolute():
                    script_full_path = project_root / script_path

                cmd = [sys.executable, str(script_full_path)]

                # Convert args to command line arguments
                for key, value in args.items():
                    if value is not None:
                        # Convert underscore to hyphen for CLI arguments
                        cli_key = key.replace('_', '-')
                        if isinstance(value, bool):
                            if value:  # Only add flag if True
                                cmd.append(f"--{cli_key}")
                        elif isinstance(value, list):
                            # Handle list arguments (e.g., input files)
                            cmd.extend([f"--{cli_key}"] + [str(v) for v in value])
                        else:
                            cmd.extend([f"--{cli_key}", str(value)])

                # Set output directory for the job
                output_dir = job_dir / "output"
                output_dir.mkdir(exist_ok=True)
                cmd.extend(["--output", str(output_dir)])

                # Run script
                log_file = job_dir / "job.log"
                env = os.environ.copy()

                with open(log_file, 'w') as log:
                    process = subprocess.Popen(
                        cmd,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        cwd=str(project_root),
                        env=env
                    )
                    self._running_jobs[job_id] = process
                    process.wait()

                # Update status based on return code
                if process.returncode == 0:
                    metadata["status"] = JobStatus.COMPLETED.value
                    # Try to find and load result file
                    self._collect_job_results(job_id, output_dir)
                else:
                    metadata["status"] = JobStatus.FAILED.value
                    metadata["error"] = f"Process exited with code {process.returncode}"

            except Exception as e:
                metadata["status"] = JobStatus.FAILED.value
                metadata["error"] = str(e)

            finally:
                metadata["completed_at"] = datetime.now().isoformat()
                self._save_metadata(job_id, metadata)
                self._running_jobs.pop(job_id, None)

        thread = threading.Thread(target=run_job, daemon=True)
        thread.start()

    def _collect_job_results(self, job_id: str, output_dir: Path):
        """Collect job results from output directory."""
        try:
            # Look for common result files
            result_files = list(output_dir.glob("*.json"))
            result_files.extend(list(output_dir.glob("*.pdb")))
            result_files.extend(list(output_dir.glob("*.log")))

            # Create a summary of outputs
            results_summary = {
                "output_directory": str(output_dir),
                "files_created": [str(f.relative_to(output_dir)) for f in result_files],
                "total_files": len(result_files)
            }

            # Save results summary
            job_dir = self.jobs_dir / job_id
            with open(job_dir / "results.json", 'w') as f:
                json.dump(results_summary, f, indent=2)

        except Exception as e:
            # Non-critical error, just log it
            pass

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a submitted job."""
        metadata = self._load_metadata(job_id)
        if not metadata:
            return {"status": "error", "error": f"Job {job_id} not found"}

        result = {
            "job_id": job_id,
            "job_name": metadata.get("job_name"),
            "status": metadata["status"],
            "submitted_at": metadata.get("submitted_at"),
            "started_at": metadata.get("started_at"),
            "completed_at": metadata.get("completed_at")
        }

        if metadata["status"] == JobStatus.FAILED.value:
            result["error"] = metadata.get("error")

        return result

    def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """Get results of a completed job."""
        metadata = self._load_metadata(job_id)
        if not metadata:
            return {"status": "error", "error": f"Job {job_id} not found"}

        if metadata["status"] != JobStatus.COMPLETED.value:
            return {
                "status": "error",
                "error": f"Job not completed. Current status: {metadata['status']}"
            }

        # Load results
        job_dir = self.jobs_dir / job_id
        results_file = job_dir / "results.json"

        if results_file.exists():
            with open(results_file) as f:
                result = json.load(f)
            return {"status": "success", "result": result, "job_id": job_id}
        else:
            # Fallback - look for any outputs
            output_dir = job_dir / "output"
            if output_dir.exists():
                files = list(output_dir.glob("*"))
                return {
                    "status": "success",
                    "result": {
                        "output_directory": str(output_dir),
                        "files_created": [str(f.relative_to(output_dir)) for f in files],
                        "message": "Job completed but no structured results found"
                    },
                    "job_id": job_id
                }
            else:
                return {"status": "error", "error": "Output files not found"}

    def get_job_log(self, job_id: str, tail: int = 50) -> Dict[str, Any]:
        """Get log output from a job."""
        job_dir = self.jobs_dir / job_id
        log_file = job_dir / "job.log"

        if not log_file.exists():
            return {"status": "error", "error": f"Log not found for job {job_id}"}

        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()

            return {
                "status": "success",
                "job_id": job_id,
                "log_lines": lines[-tail:] if tail > 0 else lines,
                "total_lines": len(lines)
            }
        except Exception as e:
            return {"status": "error", "error": f"Failed to read log: {str(e)}"}

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a running job."""
        if job_id in self._running_jobs:
            try:
                self._running_jobs[job_id].terminate()
                metadata = self._load_metadata(job_id)
                if metadata:
                    metadata["status"] = JobStatus.CANCELLED.value
                    metadata["completed_at"] = datetime.now().isoformat()
                    self._save_metadata(job_id, metadata)
                return {"status": "success", "message": f"Job {job_id} cancelled"}
            except Exception as e:
                return {"status": "error", "error": f"Failed to cancel job: {str(e)}"}

        # Check if job exists and what its status is
        metadata = self._load_metadata(job_id)
        if not metadata:
            return {"status": "error", "error": f"Job {job_id} not found"}

        if metadata["status"] in [JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
            return {"status": "error", "error": f"Job {job_id} is already {metadata['status']}"}

        return {"status": "error", "error": f"Job {job_id} not running"}

    def list_jobs(self, status: Optional[str] = None) -> Dict[str, Any]:
        """List all jobs, optionally filtered by status."""
        jobs = []
        try:
            for job_dir in self.jobs_dir.iterdir():
                if job_dir.is_dir():
                    metadata = self._load_metadata(job_dir.name)
                    if metadata:
                        if status is None or metadata["status"] == status:
                            jobs.append({
                                "job_id": metadata["job_id"],
                                "job_name": metadata.get("job_name"),
                                "status": metadata["status"],
                                "submitted_at": metadata.get("submitted_at"),
                                "script": metadata.get("script", "unknown")
                            })
        except Exception as e:
            return {"status": "error", "error": f"Failed to list jobs: {str(e)}"}

        # Sort by submission time (newest first)
        jobs.sort(key=lambda x: x.get("submitted_at", ""), reverse=True)

        return {"status": "success", "jobs": jobs, "total": len(jobs)}

    def _save_metadata(self, job_id: str, metadata: Dict):
        """Save job metadata to disk."""
        meta_file = self.jobs_dir / job_id / "metadata.json"
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save metadata for job {job_id}: {e}")

    def _load_metadata(self, job_id: str) -> Optional[Dict]:
        """Load job metadata from disk."""
        meta_file = self.jobs_dir / job_id / "metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load metadata for job {job_id}: {e}")
        return None

# Global job manager instance
job_manager = JobManager()