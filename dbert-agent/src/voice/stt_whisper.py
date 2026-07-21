import os
import subprocess
import logging
from pathlib import Path
import litellm

logger = logging.getLogger("dbert.voice.stt_whisper")

def transcribe(audio_path: str, model_size: str = "base", config_manager: os.PathLike = None, provider_manager: os.PathLike = None) -> str:
    """
    Transcribes a WAV audio file using local Whisper.cpp or OpenAI Whisper API fallback.
    """
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found at {audio_path}")
        
    logger.info(f"Initiating Speech-to-Text transcription on {audio_path}")
    
    # 1. Look for local Whisper.cpp executable
    # Check ~/.dbert/bin/whisper/main.exe first, then ~/.dbert/bin/whisper/whisper.exe
    home_dir = Path.home() / ".dbert"
    local_whisper_bin = home_dir / "bin" / "whisper" / "main.exe"
    if not local_whisper_bin.exists():
        local_whisper_bin = home_dir / "bin" / "whisper" / "whisper.exe"
        
    # Check system PATH
    whisper_path = None
    if local_whisper_bin.exists():
        whisper_path = str(local_whisper_bin)
    else:
        # Search PATH
        for path in os.environ.get("PATH", "").split(os.pathsep):
            p = Path(path)
            for name in ["whisper-cli.exe", "whisper.exe", "main.exe"]:
                test_path = p / name
                if test_path.exists():
                    whisper_path = str(test_path)
                    break
            if whisper_path:
                break
                
    if whisper_path:
        logger.info(f"Using local Whisper binary at {whisper_path}")
        model_path = home_dir / "bin" / "whisper" / f"ggml-{model_size}.bin"
        if not model_path.exists():
            # Try to find any ggml model in the folder
            whisper_folder = home_dir / "bin" / "whisper"
            models = list(whisper_folder.glob("ggml-*.bin")) if whisper_folder.exists() else []
            if models:
                model_path = models[0]
                logger.info(f"ggml-{model_size}.bin not found; using found model {model_path.name}")
            else:
                model_path = None
                
        if model_path and model_path.exists():
            # Run local whisper.cpp subprocess
            # Output format -otxt writes output to <audio_path>.txt
            txt_output = audio_file.with_suffix(".wav.txt")
            if txt_output.exists():
                txt_output.unlink()
                
            cmd = [
                whisper_path,
                "-m", str(model_path),
                "-f", str(audio_file),
                "-otxt"
            ]
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
                if txt_output.exists():
                    text = txt_output.read_text(encoding="utf-8").strip()
                    txt_output.unlink() # Cleanup
                    return text
            except Exception as e:
                logger.error(f"Local Whisper.cpp transcription failed: {e}")
                
    # 2. Cloud Fallback via OpenAI Whisper
    if provider_manager and "openai" in provider_manager.active_providers:
        openai_info = provider_manager.active_providers["openai"]
        api_key = openai_info.get("api_key")
        if api_key:
            logger.info("Local Whisper.cpp not available or failed; falling back to OpenAI Whisper API")
            try:
                with open(audio_file, "rb") as f:
                    response = litellm.transcription(
                        model="whisper-1",
                        file=f,
                        api_key=api_key
                    )
                return response.text.strip()
            except Exception as ex:
                logger.error(f"OpenAI Whisper API transcription failed: {ex}")
                
    # 3. Last fallback: return placeholder or warning
    raise FileNotFoundError(
        "No Speech-to-Text transcriber is available. To enable local STT, download a Whisper.cpp binary and model "
        "and place them at ~/.dbert/bin/whisper/main.exe and ~/.dbert/bin/whisper/ggml-base.bin respectively."
    )
