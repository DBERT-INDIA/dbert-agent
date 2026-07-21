import os
import json
import logging
from enum import Enum
from pathlib import Path

logger = logging.getLogger("dbert.tools.permissions")

class PermissionLevel(str, Enum):
    OFF = "off"
    ASK_EVERY_TIME = "ask_every_time"
    ASK_ONCE = "ask_once"
    ALWAYS_ALLOW = "always_allow"

def get_permissions_file(workspace_id: str, app_dir: Path = None) -> Path:
    if app_dir is None:
        workspace_dir = Path.home() / ".dbert" / "workspaces" / workspace_id
    else:
        workspace_dir = Path(app_dir) / "workspaces" / workspace_id
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return workspace_dir / "permissions.json"

def get_tool_permission(tool_name: str, workspace_id: str, config_manager) -> PermissionLevel:
    """
    Checks the permission level for a given tool in a workspace.
    Looks for overrides in the workspace permissions.json, otherwise falls back to the global default.
    """
    default_level_str = config_manager.config.get("permissions", {}).get("default_level", "ask_every_time")
    try:
        default_level = PermissionLevel(default_level_str)
    except ValueError:
        default_level = PermissionLevel.ASK_EVERY_TIME
        
    permissions_file = get_permissions_file(workspace_id, config_manager.app_dir)
    
    if permissions_file.exists():
        try:
            with open(permissions_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if tool_name in data:
                return PermissionLevel(data[tool_name])
        except Exception as e:
            logger.error(f"Failed to read permissions file: {e}")
            
    return default_level

def set_tool_permission(tool_name: str, level: PermissionLevel, workspace_id: str, config_manager) -> None:
    """
    Persists a permission level choice for a specific tool in the workspace.
    """
    permissions_file = get_permissions_file(workspace_id, config_manager.app_dir)
    data = {}
    if permissions_file.exists():
        try:
            with open(permissions_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read permissions before writing: {e}")
            
    data[tool_name] = level.value
    
    try:
        import tempfile
        fd, tmp = tempfile.mkstemp(dir=str(permissions_file.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, str(permissions_file))
        logger.info(f"Updated permission for tool '{tool_name}' in workspace '{workspace_id}' to '{level.value}'")
    except Exception as e:
        logger.error(f"Failed to write permissions update: {e}")
