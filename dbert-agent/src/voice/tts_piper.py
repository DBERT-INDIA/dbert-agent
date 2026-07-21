import os
import subprocess
import logging
from pathlib import Path
import tempfile
import litellm

logger = logging.getLogger("dbert.voice.tts_piper")

def synthesize(
    text: str,
    voice: str = "en_US-lessac-medium",
    config_manager: os.PathLike = None,
    provider_manager: os.PathLike = None
) -> str:
    """
    Synthesizes text into speech (WAV format) using local Piper TTS or OpenAI TTS API.
    Returns: Path to output WAV file
    """
    logger.info(f"Synthesizing text: '{text[:40]}...'")
    
    # 1. Determine temporary output path in ~/.dbert/tmp/
    home_dir = Path.home() / ".dbert"
    tmp_dir = home_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique output path
    fd, output_wav_path = tempfile.mkstemp(suffix=".wav", dir=str(tmp_dir))
    os.close(fd)
    
    # 2. Look for local Piper TTS executable
    local_piper_bin = home_dir / "bin" / "piper" / "piper.exe"
    piper_path = None
    if local_piper_bin.exists():
        piper_path = str(local_piper_bin)
    else:
        # Search PATH
        for path in os.environ.get("PATH", "").split(";"):
            p = Path(path)
            test_path = p / "piper.exe"
            if test_path.exists():
                piper_path = str(test_path)
                break
                
    if piper_path:
        logger.info(f"Using local Piper binary at {piper_path}")
        model_path = home_dir / "bin" / "piper" / f"{voice}.onnx"
        if not model_path.exists():
            # Try to find any loaded voice .onnx model in the folder
            piper_folder = home_dir / "bin" / "piper"
            models = list(piper_folder.glob("*.onnx")) if piper_folder.exists() else []
            if models:
                model_path = models[0]
                logger.info(f"Voice model {voice}.onnx not found; using found model {model_path.name}")
            else:
                model_path = None
                
        if model_path and model_path.exists():
            # Execute local Piper speech synthesis
            cmd = [
                piper_path,
                "--model", str(model_path),
                "--output_file", output_wav_path
            ]
            try:
                subprocess.run(cmd, input=text.encode("utf-8"), check=True, capture_output=True, timeout=30)
                return output_wav_path
            except Exception as e:
                logger.error(f"Local Piper TTS synthesis failed: {e}")
                
    # 3. Cloud Fallback via OpenAI TTS
    if provider_manager and "openai" in provider_manager.active_providers:
        openai_info = provider_manager.active_providers["openai"]
        api_key = openai_info.get("api_key")
        if api_key:
            logger.info("Local Piper TTS not available or failed; falling back to OpenAI TTS API")
            try:
                response = litellm.speech(
                    model="tts-1",
                    input=text,
                    voice="alloy",
                    api_key=api_key
                )
                with open(output_wav_path, "wb") as f:
                    f.write(response.content)
                return output_wav_path
            except Exception as ex:
                logger.error(f"OpenAI TTS API synthesis failed: {ex}")
                
    # 4. Fallback: raise exception
    raise FileNotFoundError(
        "No speech synthesizer is available. To enable local TTS, download a Piper executable and voice ONNX model "
        "and place them at ~/.dbert/bin/piper/piper.exe and ~/.dbert/bin/piper/en_US-lessac-medium.onnx respectively."
    )
