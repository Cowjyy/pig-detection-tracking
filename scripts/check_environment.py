"""Print environment information on the laboratory server; never install or train."""
import argparse
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-torch", action="store_true", help="Import an already installed PyTorch and inspect CUDA")
    args = parser.parse_args()
    info = dict(python=sys.version, executable=sys.executable, os=platform.platform(), packages={})
    for name in ("numpy", "Pillow", "opencv-python", "opencv-python-headless", "torch", "torchvision", "mmdet", "mmcv", "mmengine"):
        try:
            info["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][name] = None
    if shutil.which("nvidia-smi"):
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=30)
        info["nvidia_smi"] = dict(returncode=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())
    else:
        info["nvidia_smi"] = "not found"
    if args.check_torch:
        if importlib.util.find_spec("torch"):
            import torch
            info["torch_cuda"] = dict(available=torch.cuda.is_available(), compiled_cuda=torch.version.cuda,
                devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        else:
            info["torch_cuda"] = "torch not installed"
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
