import os
import yaml
import logging
from pathlib import Path
import keyring

logger = logging.getLogger("dbert.config_manager")

class ConfigManager:
    def __init__(self, app_dir: Path = None, default_config_path: Path = None):
        if app_dir is None:
            self.app_dir = Path.home() / ".dbert"
        else:
            self.app_dir = Path(app_dir)
            
        self._init_directories()
        self.config_path = self.app_dir / "config.yaml"
        
        if default_config_path is None:
            current_dir = Path(__file__).resolve().parent
            self.default_config_path = current_dir.parent.parent / "config" / "default_config.yaml"
        else:
            self.default_config_path = Path(default_config_path)
            
        self.config = self.load_config()

    def _init_directories(self) -> None:
        self.app_dir.mkdir(parents=True, exist_ok=True)
        (self.app_dir / "workspaces").mkdir(parents=True, exist_ok=True)
        (self.app_dir / "history").mkdir(parents=True, exist_ok=True)
        (self.app_dir / "monitors").mkdir(parents=True, exist_ok=True)
        (self.app_dir / "logs").mkdir(parents=True, exist_ok=True)

    def load_config(self) -> dict:
        default_config = {}
        if self.default_config_path.exists():
            try:
                with open(self.default_config_path, "r") as f:
                    default_config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"Failed to load default config from {self.default_config_path}: {e}")
        
        if not default_config:
            # Hardcoded fallback config
            default_config = {
                "providers": {
                    "lmstudio-local": {
                        "type": "local",
                        "base_url": "http://localhost:1234/v1",
                        "enabled": True
                    },
                    "openai": {
                        "type": "cloud",
                        "enabled": False,
                        "api_key_ref": "keystore:openai"
                    },
                    "anthropic": {
                        "type": "cloud",
                        "enabled": False,
                        "api_key_ref": "keystore:anthropic"
                    },
                    "gemini": {
                        "type": "cloud",
                        "enabled": False,
                        "api_key_ref": "keystore:gemini"
                    }
                },
                "default_provider_order": ["lmstudio-local", "openai", "anthropic", "gemini"],
                "ui_prefs": {
                    "theme": "dark",
                    "agent_manager_width_px": 280,
                    "font_scale": 1.0
                },
                "hardware": {
                    "auto_detect": True,
                    "whisper_model_size": "auto",
                    "piper_quality": "auto",
                    "max_parallel_jobs": "auto"
                },
                "voice": {
                    "default_mode": "text",
                    "wake_word_enabled": False
                },
                "permissions": {
                    "default_level": "ask_every_time"
                }
            }

        if not self.config_path.exists():
            self.save_config(default_config)
            return default_config
        
        try:
            with open(self.config_path, "r") as f:
                user_config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load user config from {self.config_path}, falling back to defaults: {e}")
            user_config = {}
            
        merged = self._merge_dicts(default_config, user_config)
        return merged

    def _merge_dicts(self, dict1: dict, dict2: dict) -> dict:
        result = dict1.copy()
        for k, v in dict2.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._merge_dicts(result[k], v)
            else:
                result[k] = v
        return result

    def save_config(self, config: dict = None) -> None:
        if config is not None:
            self.config = config
        try:
            with open(self.config_path, "w") as f:
                yaml.safe_dump(self.config, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_path}: {e}")

    def get_provider_key(self, provider: str) -> str | None:
        try:
            return keyring.get_password("dbert", provider)
        except Exception as e:
            logger.error(f"Error accessing keystore for provider {provider}: {e}")
            return None

    def set_provider_key(self, provider: str, key: str) -> None:
        try:
            keyring.set_password("dbert", provider, key)
        except Exception as e:
            logger.error(f"Error setting key in keystore for provider {provider}: {e}")
            raise e
