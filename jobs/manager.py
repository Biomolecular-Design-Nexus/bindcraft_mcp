"""Job manager for BindCraft MCP async operations.

Provides a simple job queue for running scripts asynchronously and tracking their status.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class JobStatus(Enum):
    """Job status states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Represents an async job."""
    job_id: str
    job_name: str
    script_path: str
    args: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    pid: Optional[int] = None
    return_code: Optional[int] = None
    error: Optional[str] = None
    log_file: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Convert job to dictionary."""
        return {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "script_path": self.script_path,
            "args": self.args,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "pid": self.pid,
            "return_code": self.return_code,
            "error": self.error,
            "log_file": self.log_file,
            "result": self.result,
        }


class JobManager:
    """Manages async job execution and tracking."""

    def __init__(self, jobs_dir: Optional[Path] = None):
        """Initialize the job manager.

        Args:
            jobs_dir: Directory to store job logs and state. Defaults to ./jobs_data
        """
        self.jobs_dir = jobs_dir or Path(__file__).parent / "jobs_data"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

        # Load existing jobs from disk
        self._load_jobs()

    def _load_jobs(self):
        """Load job state from disk."""
        jobs_file = self.jobs_dir / "jobs.json"
        if jobs_file.exists():
            try:
                with open(jobs_file, 'r') as f:
                    jobs_data = json.load(f)
                for job_data in jobs_data:
                    job = Job(
                        job_id=job_data["job_id"],
                        job_name=job_data["job_name"],
                        script_path=job_data["script_path"],
                        args=job_data["args"],
                        status=JobStatus(job_data["status"]),
                        created_at=datetime.fromisoformat(job_data["created_at"]),
                        started_at=datetime.fromisoformat(job_data["started_at"]) if job_data.get("started_at") else None,
                        completed_at=datetime.fromisoformat(job_data["completed_at"]) if job_data.get("completed_at") else None,
                        pid=job_data.get("pid"),
                        return_code=job_data.get("return_code"),
                        error=job_data.get("error"),
                        log_file=job_data.get("log_file"),
                        result=job_data.get("result"),
                    )
                    self._jobs[job.job_id] = job
            except Exception as e:
                print(f"Warning: Could not load jobs from {jobs_file}: {e}")

    def _save_jobs(self):
        """Save job state to disk."""
        jobs_file = self.jobs_dir / "jobs.json"
        jobs_data = [job.to_dict() for job in self._jobs.values()]
        with open(jobs_file, 'w') as f:
            json.dump(jobs_data, f, indent=2)

    def submit_job(
        self,
        script_path: str,
        args: Dict[str, Any],
        job_name: Optional[str] = None
    ) -> dict:
        """Submit a new job for async execution.

        Args:
            script_path: Path to the Python script to run
            args: Dictionary of arguments to pass to the script
            job_name: Optional human-readable name for the job

        Returns:
            Dictionary with job_id and submission status
        """
        # Generate job ID
        job_id = str(uuid.uuid4())[:8]
        job_name = job_name or f"job_{job_id}"

        # Validate script exists
        if not Path(script_path).exists():
            return {
                "status": "error",
                "error": f"Script not found: {script_path}"
            }

        # Create job
        job = Job(
            job_id=job_id,
            job_name=job_name,
            script_path=script_path,
            args=args,
        )

        # Create log file
        log_file = self.jobs_dir / f"{job_id}.log"
        job.log_file = str(log_file)

        # Store job
        with self._lock:
            self._jobs[job_id] = job
            self._save_jobs()

        # Start job in background thread
        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        thread.start()

        return {
            "status": "submitted",
            "job_id": job_id,
            "job_name": job_name,
            "message": f"Job submitted successfully. Use get_job_status('{job_id}') to check progress."
        }

    def _run_job(self, job_id: str):
        """Run a job in the background."""
        job = self._jobs.get(job_id)
        if not job:
            return

        try:
            # Build command
            cmd = [sys.executable, job.script_path]
            for key, value in job.args.items():
                if value is not None:
                    cmd.append(f"--{key}={value}")

            # Update status
            with self._lock:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now()
                self._save_jobs()

            # Run the command
            with open(job.log_file, 'w') as log_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                with self._lock:
                    job.pid = process.pid
                    self._save_jobs()

                # Wait for completion
                process.wait()

                with self._lock:
                    job.return_code = process.returncode
                    job.completed_at = datetime.now()

                    if process.returncode == 0:
                        job.status = JobStatus.COMPLETED
                        # Try to extract result from log
                        job.result = self._extract_result(job.log_file)
                    else:
                        job.status = JobStatus.FAILED
                        job.error = f"Process exited with code {process.returncode}"

                    self._save_jobs()

        except Exception as e:
            with self._lock:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                job.error = str(e)
                self._save_jobs()

    def _extract_result(self, log_file: str) -> Optional[dict]:
        """Try to extract result from log file."""
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()

            # Look for JSON result at end of log
            for line in reversed(lines[-50:]):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        pass
            return None
        except Exception:
            return None

    def get_job_status(self, job_id: str) -> dict:
        """Get the status of a job.

        Args:
            job_id: The job ID to check

        Returns:
            Dictionary with job status information
        """
        job = self._jobs.get(job_id)
        if not job:
            return {
                "status": "error",
                "error": f"Job not found: {job_id}"
            }

        # Check if running job is still alive
        if job.status == JobStatus.RUNNING and job.pid:
            try:
                os.kill(job.pid, 0)  # Check if process exists
            except OSError:
                # Process no longer exists
                with self._lock:
                    job.status = JobStatus.FAILED
                    job.completed_at = datetime.now()
                    job.error = "Process terminated unexpectedly"
                    self._save_jobs()

        return {
            "status": "success",
            "job_id": job.job_id,
            "job_name": job.job_name,
            "job_status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "pid": job.pid,
            "return_code": job.return_code,
            "error": job.error,
            "log_file": job.log_file,
        }

    def get_job_result(self, job_id: str) -> dict:
        """Get the result of a completed job.

        Args:
            job_id: The job ID to get results for

        Returns:
            Dictionary with job results or error
        """
        job = self._jobs.get(job_id)
        if not job:
            return {
                "status": "error",
                "error": f"Job not found: {job_id}"
            }

        if job.status == JobStatus.PENDING:
            return {
                "status": "pending",
                "message": "Job has not started yet"
            }

        if job.status == JobStatus.RUNNING:
            return {
                "status": "running",
                "message": "Job is still running"
            }

        if job.status == JobStatus.CANCELLED:
            return {
                "status": "cancelled",
                "message": "Job was cancelled"
            }

        if job.status == JobStatus.FAILED:
            return {
                "status": "failed",
                "error": job.error,
                "return_code": job.return_code
            }

        # Completed
        return {
            "status": "success",
            "job_id": job.job_id,
            "job_name": job.job_name,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "return_code": job.return_code,
            "result": job.result,
            "log_file": job.log_file,
        }

    def get_job_log(self, job_id: str, tail: int = 50) -> dict:
        """Get log output from a job.

        Args:
            job_id: The job ID to get logs for
            tail: Number of lines from end (0 for all)

        Returns:
            Dictionary with log lines
        """
        job = self._jobs.get(job_id)
        if not job:
            return {
                "status": "error",
                "error": f"Job not found: {job_id}"
            }

        if not job.log_file or not Path(job.log_file).exists():
            return {
                "status": "success",
                "job_id": job.job_id,
                "lines": [],
                "total_lines": 0,
                "message": "No log file available yet"
            }

        try:
            with open(job.log_file, 'r') as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)

            if tail > 0:
                lines = all_lines[-tail:]
            else:
                lines = all_lines

            return {
                "status": "success",
                "job_id": job.job_id,
                "lines": [line.rstrip() for line in lines],
                "total_lines": total_lines,
                "showing": len(lines),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to read log: {e}"
            }

    def cancel_job(self, job_id: str) -> dict:
        """Cancel a running job.

        Args:
            job_id: The job ID to cancel

        Returns:
            Success or error message
        """
        job = self._jobs.get(job_id)
        if not job:
            return {
                "status": "error",
                "error": f"Job not found: {job_id}"
            }

        if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
            return {
                "status": "error",
                "error": f"Cannot cancel job with status: {job.status.value}"
            }

        if job.status == JobStatus.RUNNING and job.pid:
            try:
                os.kill(job.pid, signal.SIGTERM)
                # Give it a moment then force kill if needed
                time.sleep(1)
                try:
                    os.kill(job.pid, signal.SIGKILL)
                except OSError:
                    pass  # Already dead
            except OSError as e:
                return {
                    "status": "error",
                    "error": f"Failed to kill process: {e}"
                }

        with self._lock:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            self._save_jobs()

        return {
            "status": "success",
            "message": f"Job {job_id} cancelled"
        }

    def list_jobs(self, status: Optional[str] = None) -> dict:
        """List all jobs, optionally filtered by status.

        Args:
            status: Filter by status (pending, running, completed, failed, cancelled)

        Returns:
            List of jobs
        """
        jobs = []

        for job in self._jobs.values():
            if status and job.status.value != status:
                continue

            jobs.append({
                "job_id": job.job_id,
                "job_name": job.job_name,
                "status": job.status.value,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            })

        # Sort by created_at descending
        jobs.sort(key=lambda x: x["created_at"], reverse=True)

        return {
            "status": "success",
            "jobs": jobs,
            "total": len(jobs),
            "filter": status,
        }


# Global job manager instance
job_manager = JobManager()
