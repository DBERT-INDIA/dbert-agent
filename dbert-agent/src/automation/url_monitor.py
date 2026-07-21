import re
import os
import logging
import requests
import difflib
from pathlib import Path
from typing import Dict, Any

from src.automation.scheduler import get_monitors_dir

logger = logging.getLogger("dbert.automation.url_monitor")

def clean_html_to_text(html: str) -> str:
    """Strips HTML tags, styles, and scripts to construct a plain text layout."""
    html = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def diff_snapshot(old_text: str, new_text: str) -> str:
    """Generates a standard unified diff comparison of two snapshot strings."""
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile="Previous Snapshot",
        tofile="Current Snapshot",
        lineterm=""
    )
    return "\n".join(diff)

def check_url_for_changes(job_def: Dict[str, Any], app_dir: Path = None) -> Dict[str, Any]:
    """
    Fetches the monitored URL, cleans HTML tags, and compares against the previous text snapshot.
    Saves new state and returns status details.
    """
    url = job_def.get("payload", {}).get("url")
    job_id = job_def["id"]
    if not url:
        return {"changed": False, "message": "Error: Missing monitor URL in payload."}
        
    logger.info(f"Checking URL for changes: {url}")
    headers = {"User-Agent": "Mozilla/5.0 DBERTMonitor/0.1"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return {"changed": False, "message": f"Error: Site returned status code {r.status_code}."}
            
        current_text = clean_html_to_text(r.text)
    except Exception as e:
        logger.error(f"Failed to fetch monitor URL {url}: {e}")
        return {"changed": False, "message": f"Fetch failed: {e}."}
        
    # Read previous snapshot
    mon_dir = get_monitors_dir(app_dir)
    snapshot_file = mon_dir / f"{job_id}_snapshot.txt"
    
    import tempfile
    if not snapshot_file.exists():
        # First execution, save initial snapshot
        try:
            fd, tmp = tempfile.mkstemp(dir=str(mon_dir), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(current_text)
            os.replace(tmp, str(snapshot_file))
            return {"changed": False, "message": "Initial snapshot saved."}
        except Exception as e:
            logger.error(f"Failed to save initial snapshot: {e}")
            return {"changed": False, "message": f"Failed to save initial state: {e}."}
            
    try:
        previous_text = snapshot_file.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read previous snapshot: {e}")
        previous_text = ""
        
    if current_text == previous_text:
        return {"changed": False, "message": "No changes detected."}
        
    # Content changed! Compute diff
    diff_output = diff_snapshot(previous_text, current_text)
    
    # Save new snapshot
    try:
        fd, tmp = tempfile.mkstemp(dir=str(mon_dir), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(current_text)
        os.replace(tmp, str(snapshot_file))
        
        # Save diff audit file
        diff_file = mon_dir / f"{job_id}_diff.txt"
        fd2, tmp2 = tempfile.mkstemp(dir=str(mon_dir), suffix=".tmp")
        with os.fdopen(fd2, "w", encoding="utf-8") as f:
            f.write(diff_output)
        os.replace(tmp2, str(diff_file))
    except Exception as e:
        logger.error(f"Failed to update files for changed job: {e}")
        
    return {
        "changed": True,
        "diff": diff_output,
        "message": f"Change detected! Saved diff to {job_id}_diff.txt."
    }
