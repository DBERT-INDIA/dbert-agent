import logging
import requests
from dataclasses import dataclass
from typing import List, Dict, Any

logger = logging.getLogger("dbert.provider_manager")

@dataclass
class ModelInfo:
    id: str
    provider: str      # e.g., 'lmstudio-local', 'openai', 'anthropic', 'gemini'
    is_local: bool
    context_length: int | None = None
    size: str | None = None

class ProviderManager:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.active_providers: Dict[str, Dict[str, Any]] = {}
        
        # Static registry of common cloud models
        self.cloud_models = {
            "openai": [
                {"id": "openai/gpt-4o", "context_length": 128000, "size": "Unknown"},
                {"id": "openai/gpt-4o-mini", "context_length": 128000, "size": "Unknown"},
                {"id": "openai/gpt-4-turbo", "context_length": 128000, "size": "Unknown"},
                {"id": "openai/gpt-3.5-turbo", "context_length": 16385, "size": "Unknown"}
            ],
            "anthropic": [
                {"id": "anthropic/claude-3-5-sonnet-latest", "context_length": 200000, "size": "Unknown"},
                {"id": "anthropic/claude-3-5-sonnet-20240620", "context_length": 200000, "size": "Unknown"},
                {"id": "anthropic/claude-3-haiku-20240307", "context_length": 200000, "size": "Unknown"},
                {"id": "anthropic/claude-3-opus-20240229", "context_length": 200000, "size": "Unknown"}
            ],
            "gemini": [
                {"id": "gemini/gemini-1.5-flash", "context_length": 1048576, "size": "Unknown"},
                {"id": "gemini/gemini-1.5-pro", "context_length": 2097152, "size": "Unknown"},
                {"id": "gemini/gemini-1.0-pro", "context_length": 30720, "size": "Unknown"}
            ]
        }

    def register_local_provider(self) -> None:
        providers_cfg = self.config_manager.config.get("providers", {})
        lmstudio_cfg = providers_cfg.get("lmstudio-local", {})
        
        if lmstudio_cfg.get("enabled", True):
            self.active_providers["lmstudio-local"] = {
                "type": "local",
                "base_url": lmstudio_cfg.get("base_url", "http://localhost:1234/v1"),
                "api_key": "lm-studio"
            }
            logger.info("Registered local LM Studio provider.")

        # Check for optional ollama provider if enabled in config
        if "ollama" in providers_cfg:
            ollama_cfg = providers_cfg.get("ollama", {})
            if ollama_cfg.get("enabled", False):
                self.active_providers["ollama"] = {
                    "type": "local",
                    "base_url": ollama_cfg.get("base_url", "http://localhost:11434/v1"),
                    "api_key": "ollama"
                }
                logger.info("Registered local Ollama provider.")

    def register_cloud_providers(self) -> None:
        providers_cfg = self.config_manager.config.get("providers", {})
        for name, p_cfg in providers_cfg.items():
            if p_cfg.get("type") == "cloud" and p_cfg.get("enabled", False):
                api_key = self.config_manager.get_provider_key(name)
                if api_key:
                    self.active_providers[name] = {
                        "type": "cloud",
                        "api_key": api_key,
                        "base_url": p_cfg.get("base_url")
                    }
                    logger.info(f"Registered cloud provider: {name}")
                else:
                    logger.warning(f"Cloud provider {name} is enabled, but no API key was found in the secure keystore.")

    def register_cloud_provider(self, name: str, api_key: str, base_url: str | None = None) -> None:
        self.active_providers[name] = {
            "type": "cloud",
            "api_key": api_key,
            "base_url": base_url
        }
        self.config_manager.set_provider_key(name, api_key)
        
        if "providers" not in self.config_manager.config:
            self.config_manager.config["providers"] = {}
            
        self.config_manager.config["providers"][name] = {
            "type": "cloud",
            "enabled": True,
            "api_key_ref": f"keystore:{name}"
        }
        if base_url:
            self.config_manager.config["providers"][name]["base_url"] = base_url
            
        self.config_manager.save_config()
        logger.info(f"Successfully registered and saved key for cloud provider: {name}")

    def remove_provider(self, name: str) -> None:
        if name in self.active_providers:
            del self.active_providers[name]
        
        if name in self.config_manager.config.get("providers", {}):
            self.config_manager.config["providers"][name]["enabled"] = False
            self.config_manager.save_config()
            
        try:
            import keyring
            keyring.delete_password("dbert", name)
            logger.info(f"Removed API key for provider: {name}")
        except Exception as e:
            logger.warning(f"Could not delete API key from keystore for {name}: {e}")

    def list_available_models(self) -> List[ModelInfo]:
        models = []
        for name, info in self.active_providers.items():
            if info["type"] == "local":
                base_url = info["base_url"]
                url = f"{base_url.rstrip('/')}/models"
                try:
                    res = requests.get(url, timeout=5.0)
                    if res.status_code == 200:
                        data = res.json()
                        for model_item in data.get("data", []):
                            model_id = model_item.get("id")
                            models.append(ModelInfo(
                                id=model_id,
                                provider=name,
                                is_local=True,
                                context_length=model_item.get("context_length"),
                                size=model_item.get("size")
                            ))
                except Exception as e:
                    logger.warning(f"Could not retrieve models from local provider {name} at {url}: {e}")
            else:
                api_key = info.get("api_key")
                if api_key:
                    provider_models = self.cloud_models.get(name, [])
                    for m in provider_models:
                        models.append(ModelInfo(
                            id=m["id"],
                            provider=name,
                            is_local=False,
                            context_length=m.get("context_length"),
                            size=m.get("size")
                        ))
        return models

    def test_provider_connection(self, name: str) -> bool:
        info = self.active_providers.get(name)
        if not info:
            logger.error(f"Provider {name} is not registered or active.")
            return False
            
        if info["type"] == "local":
            base_url = info["base_url"]
            try:
                res = requests.get(f"{base_url.rstrip('/')}/models", timeout=5.0)
                return res.status_code == 200
            except Exception:
                return False
        else:
            api_key = info.get("api_key")
            if not api_key:
                return False
            
            test_models = self.cloud_models.get(name, [])
            if not test_models:
                return False
            model_id = test_models[0]["id"]
            
            try:
                # Turn off telemetry and billing alerts during test completion
                import litellm
                litellm.completion(
                    model=model_id,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                    api_key=api_key
                )
                return True
            except Exception as e:
                logger.error(f"API connection test for {name} failed: {e}")
                return False

    def list_active_providers(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        for name, info in self.active_providers.items():
            api_key = info.get("api_key")
            masked = self._mask_key(api_key) if info["type"] == "cloud" else "N/A (Local)"
            # Test connection
            is_connected = self.test_provider_connection(name)
            result[name] = {
                "type": info["type"],
                "base_url": info.get("base_url"),
                "key_status": "Loaded" if api_key else "Missing",
                "masked_key": masked,
                "connection_ok": is_connected
            }
        return result

    def _mask_key(self, key: str | None) -> str:
        if not key:
            return "None"
        if len(key) <= 8:
            return "****"
        return f"{key[:4]}...{key[-4:]}"
