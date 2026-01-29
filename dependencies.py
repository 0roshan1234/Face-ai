
import sys
import subprocess
from importlib.metadata import version, PackageNotFoundError

# Define required packages with minimum versions
REQUIRED_PACKAGES = {
    # Core
    "torch": "2.0.0",
    "torchvision": "0.15.0",
    "numpy": "1.24.0",
    
    # DECA dependencies
    "face_alignment": None,  # Any version
    "chumpy": None,
    "yacs": None,
    "kornia": None,
    "scikit-image": None,
    "scipy": None,
    
    # MagicFace dependencies
    "diffusers": "0.21.0",
    "transformers": "4.30.0",
    "accelerate": None,
    "safetensors": None,
    
    # InsightFace dependencies
    "insightface": None,
    "onnxruntime": None,  # or onnxruntime-gpu
    "opencv-python": None,
    
    # Additional
    "Pillow": None,
    "tqdm": None,
    "huggingface_hub": None,
}

# Alternate package names (import name -> pip name)
IMPORT_TO_PIP = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "skimage": "scikit-image",
    "yaml": "PyYAML",
    "face_alignment": "face-alignment",
}

def get_installed_version(package_name):
    """Get installed version of a package"""
    try:
        return version(package_name)
    except PackageNotFoundError:
        # Try alternate names
        alt_names = {
            "opencv-python": ["opencv-python-headless", "opencv-contrib-python"],
            "onnxruntime": ["onnxruntime-gpu"],
        }
        for alt in alt_names.get(package_name, []):
            try:
                return version(alt) + f" ({alt})"
            except PackageNotFoundError:
                continue
        return None

def compare_versions(installed, required):
    """Compare version strings (simple comparison)"""
    if required is None:
        return True  # Any version is OK
    
    try:
        inst_parts = [int(x) for x in installed.split(".")[:3]]
        req_parts = [int(x) for x in required.split(".")[:3]]
        return inst_parts >= req_parts
    except:
        return True  # Can't compare, assume OK

def check_import(module_name):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            return f"CUDA {torch.version.cuda} - {torch.cuda.get_device_name(0)}"
        else:
            return "NOT AVAILABLE"
    except:
        return "TORCH NOT INSTALLED"

def main():
    print("=" * 60)
    print("DEPENDENCY CHECKER - MagicFace + DECA Pipeline")
    print("=" * 60)
    print()
    
    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python Version: {py_version}")
    if sys.version_info < (3, 9):
        print("  [WARNING] Python 3.9+ recommended")
    print()
    
    # Check CUDA
    print(f"CUDA Status: {check_cuda()}")
    print()
    
    # Check packages
    print("-" * 60)
    print(f"{'Package':<25} {'Status':<10} {'Installed':<15} {'Required':<10}")
    print("-" * 60)
    
    missing = []
    version_mismatch = []
    ok_packages = []
    
    for package, min_version in REQUIRED_PACKAGES.items():
        installed = get_installed_version(package)
        
        if installed is None:
            status = "[MISSING]"
            missing.append(package)
        elif min_version and not compare_versions(installed.split()[0], min_version):
            status = "[OLD]"
            version_mismatch.append((package, installed, min_version))
        else:
            status = "[OK]"
            ok_packages.append(package)
        
        req_str = min_version or "any"
        inst_str = installed or "not found"
        print(f"{package:<25} {status:<10} {inst_str:<15} {req_str:<10}")
    
    print("-" * 60)
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  OK: {len(ok_packages)}")
    print(f"  Missing: {len(missing)}")
    print(f"  Version mismatch: {len(version_mismatch)}")
    print()
    
    # Generate install commands
    if missing or version_mismatch:
        print("=" * 60)
        print("FIX COMMANDS")
        print("=" * 60)
        print()
        
        if missing:
            print("# Install missing packages:")
            # Group by category
            core = [p for p in missing if p in ["torch", "torchvision", "numpy"]]
            deca = [p for p in missing if p in ["face_alignment", "chumpy", "yacs", "kornia", "scikit-image", "scipy"]]
            magicface = [p for p in missing if p in ["diffusers", "transformers", "accelerate", "safetensors"]]
            insightface = [p for p in missing if p in ["insightface", "onnxruntime", "opencv-python"]]
            other = [p for p in missing if p not in core + deca + magicface + insightface]
            
            if core:
                if "torch" in core:
                    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
                else:
                    print(f"pip install {' '.join(core)}")
            
            if deca:
                pip_names = [IMPORT_TO_PIP.get(p, p) for p in deca]
                print(f"pip install {' '.join(pip_names)}")
            
            if magicface:
                print(f"pip install {' '.join(magicface)}")
            
            if insightface:
                pip_names = ["onnxruntime-gpu" if p == "onnxruntime" else p for p in insightface]
                pip_names = ["opencv-python-headless" if p == "opencv-python" else p for p in pip_names]
                print(f"pip install {' '.join(pip_names)}")
            
            if other:
                print(f"pip install {' '.join(other)}")
            print()
        
        if version_mismatch:
            print("# Upgrade outdated packages:")
            for pkg, installed, required in version_mismatch:
                print(f"pip install --upgrade {pkg}>={required}")
            print()
    
    # Check critical imports
    print("=" * 60)
    print("IMPORT TEST")
    print("=" * 60)
    
    critical_imports = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("diffusers", "Diffusers"),
        ("insightface", "InsightFace"),
        ("face_alignment", "Face Alignment"),
    ]
    
    for module, name in critical_imports:
        if check_import(module):
            print(f"  [OK] {name}")
        else:
            print(f"  [FAIL] {name} - Cannot import '{module}'")
    
    print()
    
    # Save report
    report_file = "dependency_report.txt"
    with open(report_file, "w") as f:
        f.write("MagicFace + DECA Dependency Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Python: {py_version}\n")
        f.write(f"CUDA: {check_cuda()}\n\n")
        f.write(f"OK: {len(ok_packages)}\n")
        f.write(f"Missing: {missing}\n")
        f.write(f"Version Mismatch: {version_mismatch}\n")
    
    print(f"Report saved to: {report_file}")
    print()
    
    if missing or version_mismatch:
        print("[!] Please run the fix commands above to resolve issues.")
        return 1
    else:
        print("[OK] All dependencies are installed correctly!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
