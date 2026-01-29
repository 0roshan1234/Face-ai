"""
Model Download Script for Lightning AI Studio
Downloads: CodeFormer, antelopev2, buffalo_l, InstantID, inswapper
"""
import os
import subprocess
import sys

# Install gdown if not available
try:
    import gdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
    import gdown

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
    from huggingface_hub import hf_hub_download, snapshot_download


def create_dirs():
    """Create model directories."""
    dirs = [
        "models",
        "models/antelopev2",
        "models/buffalo_l", 
        "models/CodeFormer",
        "models/InstantID",
        "models/InstantID/ControlNetModel",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created: {d}")


def download_insightface_models():
    """Download antelopev2 and buffalo_l from InsightFace."""
    print("\n" + "="*60)
    print("Downloading InsightFace Models (antelopev2, buffalo_l)")
    print("="*60)
    
    # antelopev2 files
    antelope_files = [
        "1k3d68.onnx",
        "2d106det.onnx", 
        "genderage.onnx",
        "glintr100.onnx",
        "scrfd_10g_bnkps.onnx"
    ]
    
    # buffalo_l files (same structure)
    buffalo_files = [
        "1k3d68.onnx",
        "2d106det.onnx",
        "det_10g.onnx",
        "genderage.onnx",
        "w600k_r50.onnx"
    ]
    
    print("\nDownloading antelopev2...")
    for fname in antelope_files:
        try:
            path = hf_hub_download(
                repo_id="DIAMONIK7777/antelopev2",
                filename=fname,
                local_dir="models/antelopev2"
            )
            print(f"  ✓ {fname}")
        except Exception as e:
            print(f"  ✗ {fname}: {e}")
    
    print("\nDownloading buffalo_l...")
    for fname in buffalo_files:
        try:
            path = hf_hub_download(
                repo_id="DIAMONIK7777/buffalo_l",
                filename=fname,
                local_dir="models/buffalo_l"
            )
            print(f"  ✓ {fname}")
        except Exception as e:
            print(f"  ✗ {fname}: {e}")


def download_codeformer():
    """Download CodeFormer model."""
    print("\n" + "="*60)
    print("Downloading CodeFormer")
    print("="*60)
    
    try:
        # CodeFormer from GitHub releases via gdown
        url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
        output = "models/CodeFormer/codeformer.pth"
        
        if not os.path.exists(output):
            print(f"Downloading from GitHub...")
            subprocess.run(["wget", "-O", output, url], check=True)
            print(f"  ✓ codeformer.pth")
        else:
            print(f"  ✓ codeformer.pth (already exists)")
            
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print("  Try manual download from: https://github.com/sczhou/CodeFormer/releases")


def download_inswapper():
    """Download inswapper_128.onnx for face swapping."""
    print("\n" + "="*60)
    print("Downloading Inswapper (Face Swap Model)")
    print("="*60)
    
    output = "models/inswapper_128.onnx"
    
    if os.path.exists(output):
        print(f"  ✓ inswapper_128.onnx (already exists)")
        return
    
    try:
        # Try HuggingFace
        path = hf_hub_download(
            repo_id="thebiglaskowski/inswapper_128.onnx",
            filename="inswapper_128.onnx",
            local_dir="models"
        )
        print(f"  ✓ inswapper_128.onnx")
    except Exception as e:
        print(f"  ✗ HuggingFace failed: {e}")
        print("  Try manual download from: https://huggingface.co/thebiglaskowski/inswapper_128.onnx")


def download_instantid():
    """Download InstantID models."""
    print("\n" + "="*60)
    print("Downloading InstantID Models")
    print("="*60)
    
    try:
        # IP-Adapter
        print("Downloading ip-adapter.bin...")
        path = hf_hub_download(
            repo_id="InstantX/InstantID",
            filename="ip-adapter.bin",
            local_dir="models/InstantID"
        )
        print(f"  ✓ ip-adapter.bin")
        
        # ControlNet
        print("Downloading ControlNetModel...")
        snapshot_download(
            repo_id="InstantX/InstantID",
            allow_patterns=["ControlNetModel/*"],
            local_dir="models/InstantID"
        )
        print(f"  ✓ ControlNetModel/")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print("  Try: https://huggingface.co/InstantX/InstantID")


def download_magicface():
    """Download MagicFace models."""
    print("\n" + "="*60)
    print("Downloading MagicFace Models")
    print("="*60)
    
    try:
        print("Downloading denoising_unet and ID_enc...")
        snapshot_download(
            repo_id="mengtingwei/magicface",
            local_dir="MagicFace",
            ignore_patterns=["*.md", "*.txt"]
        )
        print(f"  ✓ MagicFace models downloaded")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print("  Try: https://huggingface.co/mengtingwei/magicface")


def main():
    print("="*60)
    print("    MODEL DOWNLOAD SCRIPT FOR LIGHTNING AI STUDIO")
    print("="*60)
    
    create_dirs()
    
    # Download each model set
    download_insightface_models()
    download_codeformer()
    download_inswapper()
    download_instantid()
    download_magicface()
    
    print("\n" + "="*60)
    print("    DOWNLOAD COMPLETE!")
    print("="*60)
    print("\nModel locations:")
    print("  - InsightFace: models/antelopev2/, models/buffalo_l/")
    print("  - CodeFormer: models/CodeFormer/codeformer.pth")
    print("  - Inswapper: models/inswapper_128.onnx")
    print("  - InstantID: models/InstantID/")
    print("  - MagicFace: MagicFace/denoising_unet/, MagicFace/ID_enc/")


if __name__ == "__main__":
    main()
