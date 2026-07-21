import os
import json
import time
import logging
import uuid
import subprocess
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger("dbert.automation.scheduler")

def get_monitors_dir(app_dir: Path = None) -> Path:
    if app_dir is None:
        mon_dir = Path.home() / ".dbert" / "monitors"
    else:
        mon_dir = Path(app_dir) / "monitors"
    mon_dir.mkdir(parents=True, exist_ok=True)
    return mon_dir

def schedule_job(job_def: Dict[str, Any], app_dir: Path = None) -> str:
    """Saves or updates a job definition file inside ~/.dbert/monitors/."""
    mon_dir = get_monitors_dir(app_dir)
    job_id = job_def.get("id") or str(uuid.uuid4())[:8]
    job_def["id"] = job_id
    
    # Fill defaults
    job_def.setdefault("status", "idle")
    job_def.setdefault("last_run", 0.0)
    job_def.setdefault("last_result", "")
    if "next_run" not in job_def:
        job_def["next_run"] = time.time() + job_def.get("interval_seconds", 60)
        
    import tempfile
    job_file = mon_dir / f"{job_id}.json"
    try:
        temp_fd, temp_path = tempfile.mkstemp(dir=str(mon_dir), suffix=".tmp")
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            json.dump(job_def, f, indent=2)
        os.replace(temp_path, str(job_file))
        logger.info(f"Scheduled job: {job_def['name']} (ID: {job_id})")
        return job_id
    except Exception as e:
        logger.error(f"Failed to save job definition for {job_id}: {e}")
        return ""

def list_jobs(app_dir: Path = None) -> List[Dict[str, Any]]:
    """Loads all scheduled job definitions from ~/.dbert/monitors/."""
    mon_dir = get_monitors_dir(app_dir)
    jobs = []
    for path in mon_dir.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                jobs.append(json.load(f))
        except Exception as e:
            logger.error(f"Failed to read job file {path.name}: {e}")
    return jobs

def remove_job(job_id: str, app_dir: Path = None) -> bool:
    """Deletes a scheduled job file."""
    mon_dir = get_monitors_dir(app_dir)
    job_file = mon_dir / f"{job_id}.json"
    if job_file.exists():
        try:
            job_file.unlink()
            logger.info(f"Removed scheduled job ID: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete job file {job_id}: {e}")
    return False

def send_windows_notification(title: str, message: str) -> None:
    """Natively triggers a system tray bubble balloon notification on Windows."""
    import base64
    
    ps_cmd = f"""
[void] [System.Reflection.Assembly]::LoadWithPartialName("System.Windows.Forms");
$objNotifyIcon = New-Object System.Windows.Forms.NotifyIcon;
$objNotifyIcon.Icon = [System.Drawing.SystemIcons]::Information;
$objNotifyIcon.BalloonTipIcon = "Info";
$objNotifyIcon.BalloonTipTitle = "{title.replace('"', '""')}";
$objNotifyIcon.BalloonTipText = "{message.replace('"', '""')}";
$objNotifyIcon.Visible = $True;
$objNotifyIcon.ShowBalloonTip(5000);
"""
    encoded = base64.b64encode(ps_cmd.encode('utf-16-le')).decode('utf-8')
    try:
        subprocess.Popen(["powershell", "-EncodedCommand", encoded], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Triggered Windows balloon notification: {title}")
    except Exception as e:
        logger.warning(f"Failed to send Windows desktop notification: {e}")

def run_pending_jobs(config_manager: Any, provider_manager: Any, active_model: Any) -> None:
    """
    Scans for due jobs and executes them.
    Supports 'url_monitor' and 'agent_task' job types.
    """
    jobs = list_jobs(config_manager.app_dir)
    now = time.time()
    
    for job in jobs:
        if now >= job.get("next_run", 0.0) and job.get("status") != "running":
            job_id = job["id"]
            logger.info(f"Executing due job: {job['name']} ({job_id})")
            
            # Lock state
            job["status"] = "running"
            schedule_job(job, config_manager.app_dir)
            
            try:
                if job["type"] == "url_monitor":
                    from src.automation.url_monitor import check_url_for_changes
                    res = check_url_for_changes(job, config_manager.app_dir)
                    job["last_result"] = res.get("message", "Checked.")
                    
                    if res.get("changed"):
                        # Alert user
                        msg = f"Change detected on monitored site '{job['name']}'!"
                        logger.warning(msg)
                        send_windows_notification(f"DBERT Monitor — {job['name']}", msg)
                        
                elif job["type"] == "agent_task":
                    # Run recurring AI prompt tasks
                    prompt = job["payload"].get("prompt", "")
                    
                    # Create messages context structure
                    messages = [{"role": "user", "content": prompt}]
                    from src.main import execute_completion_with_fallback
                    
                    reply, _ = execute_completion_with_fallback(
                        active_model,
                        messages,
                        provider_manager,
                        config_manager,
                        workspace_id="automation",
                        permission_callback=lambda t, a: "deny"
                    )
                    
                    job["last_result"] = f"Success: {reply[:100]}..."
                    send_windows_notification(f"DBERT Task — {job['name']}", "Recurring scheduled task completed successfully.")
                    
            except Exception as e:
                logger.error(f"Error running job {job_id}: {e}")
                job["last_result"] = f"Error: {e}"
                
            # Update timestamps
            job["last_run"] = time.time()
            job["next_run"] = time.time() + job.get("interval_seconds", 60)
            job["status"] = "idle"
            
            # Save final execution state
            schedule_job(job, config_manager.app_dir)
