import subprocess
import sys
import os

def check_nvidia_gpu():
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_cuda():
    print("PyTorch CUDA 12.4 Upgrade Tool")
    print("-------------------------------")

    if not check_nvidia_gpu():
        print("Error: No NVIDIA GPU detected. Please ensure your NVIDIA drivers are installed.")
        return

    print("Step 1: Uninstalling existing PyTorch components (to avoid conflicts)...")
    uninstall_cmd = [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"]
    subprocess.run(uninstall_cmd)

    print("\nStep 2: Installing CUDA 12.4 enabled PyTorch...")
    print("Note: This is a ~2GB download and may take several minutes depending on your connection.")
    # Official PyTorch command for CUDA 12.4 on Windows
    install_cmd = [
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu124"
    ]

    try:
        subprocess.check_call(install_cmd)
        print("\nSuccess: PyTorch with CUDA support has been installed!")
        print("-------------------------------")
        print("Verification:")
        verify_cmd = [sys.executable, "-c", "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"]
        subprocess.run(verify_cmd)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Installation failed with code {e.returncode}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    install_cuda()
