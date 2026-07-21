from typing import List, Callable
import logging
from src.core.provider_manager import ModelInfo, ProviderManager

logger = logging.getLogger("dbert.model_registry")

class NoModelsAvailableError(Exception):
    """Exception raised when no local or cloud models are available to the agent."""
    pass

def discover_models(provider_manager: ProviderManager) -> List[ModelInfo]:
    """Scans and retrieves all available models from active providers."""
    return provider_manager.list_available_models()

def resolve_model_choice(
    models: List[ModelInfo],
    cli_flag: str | None = None,
    saved_pref: str | None = None,
    prompt_callback: Callable[[List[ModelInfo]], ModelInfo] | None = None
) -> ModelInfo:
    """
    Resolves which model should be used based on configuration, CLI parameters, and availability.
    If multiple models are available and no preferences are set, prompts the user.
    """
    if not models:
        raise NoModelsAvailableError(
            "No models detected! Please ensure LM Studio is running and has a model loaded, "
            "or configure cloud API keys."
        )
    
    # 1. Resolve via CLI flag override
    if cli_flag:
        for model in models:
            if model.id == cli_flag:
                logger.info(f"Model resolved via CLI flag: {model.id}")
                return model
        logger.warning(f"CLI flag model '{cli_flag}' not found in available models.")

    # 2. Resolve via saved preference in config
    if saved_pref:
        for model in models:
            if model.id == saved_pref:
                logger.info(f"Model resolved via saved preference: {model.id}")
                return model
        logger.warning(f"Saved preference model '{saved_pref}' not found in available models.")

    # 3. Auto-select if only 1 model is available
    if len(models) == 1:
        logger.info(f"Only one model available. Auto-selecting: {models[0].id}")
        return models[0]

    # 4. Interactive prompt
    if prompt_callback:
        return prompt_callback(models)
    else:
        return prompt_user_for_model(models)

def prompt_user_for_model(models: List[ModelInfo]) -> ModelInfo:
    """CLI implementation of model picker. Prompts user via console input."""
    # List local models first, then cloud models
    local_models = [m for m in models if m.is_local]
    cloud_models = [m for m in models if not m.is_local]
    
    sorted_models = local_models + cloud_models
    
    print("\n[DBERT] Multiple models detected. Please choose one:")
    for idx, model in enumerate(sorted_models, 1):
        loc_str = "Local" if model.is_local else "Cloud"
        size_str = f" ({model.size})" if model.size and model.size != "Unknown" else ""
        print(f"  [{idx}] {model.id} [{model.provider} | {loc_str}]{size_str}")
        
    while True:
        try:
            choice = input(f"Select model (1-{len(sorted_models)}): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(sorted_models):
                selected = sorted_models[choice_idx]
                return selected
        except (ValueError, IndexError):
            pass
        print(f"Invalid input. Please enter a number between 1 and {len(sorted_models)}.")
