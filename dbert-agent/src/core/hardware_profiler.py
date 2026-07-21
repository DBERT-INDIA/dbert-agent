import os
import sys
import json
import logging
import platform
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
import psutil
import yaml

# Import ctypes only on Windows
if platform.system() == "Windows":
    import ctypes
else:
    ctypes = None

logger = logging.getLogger("dbert.hardware_profiler")

@dataclass
class HardwareProfile:
    cpu_cores: int
    ram_gb: float
    gpu_vendor: str
    gpu_vram_gb: float
    os_version: str

@dataclass
class DisplayProfile:
    width: int
    height: int
    dpi_scale: float

@dataclass
class AdaptiveDefaults:
    whisper_model_size: str
    piper_quality: str
    embedding_batch_size: int
    max_parallel_jobs: int

class HardwareProfiler:
    def __init__(self, app_dir: Path = None):
        if app_dir is None:
            self.app_dir = Path.home() / ".dbert"
        else:
            self.app_dir = Path(app_dir)
            
        self.app_dir.mkdir(parents=True, exist_ok=True)
        self.profile_path = self.app_dir / "hardware_profile.yaml"

    def detect_hardware(self) -> HardwareProfile:
        cpu_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 4
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        os_version = f"{platform.system()} {platform.release()} (v{platform.version()})"
        
        gpu_vendor = "None"
        gpu_vram_gb = 0.0
        
        if platform.system() == "Windows":
            # Attempt to query GPU details via PowerShell Win32_VideoController
            cmd = [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-Command",
                "Get-CimInstance Win32_VideoController | ForEach-Object { [PSCustomObject]@{Name=$_.Name; AdapterRAM=$_.AdapterRAM} } | ConvertTo-Json"
            ]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
                stdout = res.stdout.strip()
                if stdout:
                    data = json.loads(stdout)
                    if isinstance(data, dict):
                        data = [data]
                    
                    max_ram = -1.0
                    best_gpu = "None"
                    for item in data:
                        name = item.get("Name", "")
                        ram = item.get("AdapterRAM")
                        # Skip virtual/software renderers if actual GPU is present
                        if "citrix" in name.lower() or "remote display" in name.lower():
                            continue
                        
                        ram_val = 0.0
                        if ram is not None:
                            # Convert bytes to GB
                            ram_val = float(ram) / (1024 ** 3)
                            # WMI can sometimes return negative numbers or overflow values for RAM, handle it
                            if ram_val < 0:
                                ram_val = (float(ram) + 2**32) / (1024 ** 3)
                        
                        if ram_val > max_ram:
                            max_ram = ram_val
                            best_gpu = name
                    
                    gpu_vram_gb = round(max_ram if max_ram >= 0 else 0.0, 2)
                    
                    # Resolve vendor
                    best_gpu_lower = best_gpu.lower()
                    if "nvidia" in best_gpu_lower:
                        gpu_vendor = "NVIDIA"
                    elif "amd" in best_gpu_lower or "radeon" in best_gpu_lower:
                        gpu_vendor = "AMD"
                    elif "intel" in best_gpu_lower:
                        gpu_vendor = "Intel"
                    else:
                        gpu_vendor = best_gpu
            except Exception as e:
                logger.warning(f"PowerShell GPU detection failed, trying fallback: {e}")
                # Fallback to wmic VideoController
                try:
                    res = subprocess.run(
                        "wmic path win32_VideoController get Name,AdapterRAM /value",
                        shell=True, capture_output=True, text=True, timeout=3
                    )
                    lines = res.stdout.strip().split("\n")
                    name = ""
                    ram = ""
                    for line in lines:
                        if line.startswith("Name="):
                            name = line.split("Name=")[1].strip()
                        elif line.startswith("AdapterRAM="):
                            ram = line.split("AdapterRAM=")[1].strip()
                            
                    if name:
                        gpu_vram_gb = round(float(ram) / (1024 ** 3), 2) if ram.isdigit() else 0.0
                        name_lower = name.lower()
                        if "nvidia" in name_lower:
                            gpu_vendor = "NVIDIA"
                        elif "amd" in name_lower or "radeon" in name_lower:
                            gpu_vendor = "AMD"
                        elif "intel" in name_lower:
                            gpu_vendor = "Intel"
                        else:
                            gpu_vendor = name
                except Exception as ex:
                    logger.error(f"Fallback GPU detection failed: {ex}")
                    gpu_vendor = "None"
                    gpu_vram_gb = 0.0

        return HardwareProfile(
            cpu_cores=cpu_cores,
            ram_gb=ram_gb,
            gpu_vendor=gpu_vendor,
            gpu_vram_gb=gpu_vram_gb,
            os_version=os_version
        )

    def detect_display(self) -> DisplayProfile:
        width = 1920
        height = 1080
        dpi_scale = 1.0
        
        if platform.system() == "Windows" and ctypes is not None:
            try:
                # Setup DPI Awareness
                try:
                    ctypes.windll.shcore.SetProcessDpiAwareness(2) # PROCESS_PER_MONITOR_DPI_AWARE
                except Exception:
                    try:
                        ctypes.windll.user32.SetProcessDPIAware()
                    except Exception:
                        pass
                
                width = ctypes.windll.user32.GetSystemMetrics(0)
                height = ctypes.windll.user32.GetSystemMetrics(1)
                
                hdc = ctypes.windll.user32.GetDC(0)
                # LOGPIXELSX = 88
                dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)
                ctypes.windll.user32.ReleaseDC(0, hdc)
                
                # 96 DPI is standard 100% scale
                dpi_scale = round(dpi / 96.0, 2)
            except Exception as e:
                logger.error(f"DPI/display metrics detection failed: {e}")
                
        return DisplayProfile(width=width, height=height, dpi_scale=dpi_scale)

    def recommend_defaults(self, hw: HardwareProfile) -> AdaptiveDefaults:
        # Rules based on CPU/RAM/GPU capabilities
        if hw.ram_gb < 8.0:
            whisper_model_size = "tiny"
            piper_quality = "low"
            embedding_batch_size = 16
            max_parallel_jobs = 1
        elif hw.ram_gb < 16.0:
            whisper_model_size = "base"
            piper_quality = "medium"
            embedding_batch_size = 32
            max_parallel_jobs = 2
        else:
            # High-end profile
            # If we have an Nvidia GPU with good VRAM, we can run bigger models
            whisper_model_size = "small"
            piper_quality = "high"
            embedding_batch_size = 64
            max_parallel_jobs = 4
            
        # Adjust concurrency if CPU cores are low
        if hw.cpu_cores < 4:
            max_parallel_jobs = min(max_parallel_jobs, 2)

        return AdaptiveDefaults(
            whisper_model_size=whisper_model_size,
            piper_quality=piper_quality,
            embedding_batch_size=embedding_batch_size,
            max_parallel_jobs=max_parallel_jobs
        )

    def profile_has_changed(self, cached: HardwareProfile, current: HardwareProfile) -> bool:
        # Check if core components like CPU count, RAM, or GPU vendor/VRAM changed significantly
        if cached.cpu_cores != current.cpu_cores:
            return True
        if abs(cached.ram_gb - current.ram_gb) > 1.0: # allow minor fluctuation
            return True
        if cached.gpu_vendor != current.gpu_vendor:
            return True
        if abs(cached.gpu_vram_gb - current.gpu_vram_gb) > 0.5:
            return True
        return False

    def run_profiling(self) -> tuple[HardwareProfile, DisplayProfile, AdaptiveDefaults]:
        current_hw = self.detect_hardware()
        display = self.detect_display()
        defaults = self.recommend_defaults(current_hw)
        
        # Check if we should update cache
        should_write = True
        if self.profile_path.exists():
            try:
                with open(self.profile_path, "r") as f:
                    cached_data = yaml.safe_load(f)
                if cached_data and "hardware" in cached_data:
                    cached_hw = HardwareProfile(**cached_data["hardware"])
                    if not self.profile_has_changed(cached_hw, current_hw):
                        should_write = False
            except Exception as e:
                logger.warning(f"Could not read cached hardware profile: {e}")
        
        if should_write:
            try:
                output_data = {
                    "hardware": asdict(current_hw),
                    "display": asdict(display),
                    "defaults": asdict(defaults)
                }
                with open(self.profile_path, "w") as f:
                    yaml.safe_dump(output_data, f, default_flow_style=False)
                logger.info(f"Hardware profile saved to {self.profile_path}")
            except Exception as e:
                logger.error(f"Failed to write hardware profile to {self.profile_path}: {e}")
                
        return current_hw, display, defaults
