#!/usr/bin/env python3
"""
Download DECA for Lightning AI
Downloads DECA code from GitHub and required model files
"""

import os
import subprocess
import sys
import urllib.request
import zipfile
import gdown

def run_cmd(cmd):
    """Run a shell command"""
    print(f">> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.returncode == 0

def download_file(url, dest):
    """Download a file with progress"""
    print(f"Downloading: {url}")
    print(f"To: {dest}")
    
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[OK] Downloaded: {os.path.basename(dest)}")
        return True
    except Exception as e:
        print(f"[FAIL] Download failed: {e}")
        return False

def main():
    print("=" * 60)
    print("DECA DOWNLOADER FOR LIGHTNING AI")
    print("=" * 60)
    print()
    
    # Base directory
    base_dir = "/teamspace/studios/this_studio/Magicface"
    os.makedirs(base_dir, exist_ok=True)
    os.chdir(base_dir)
    
    deca_dir = os.path.join(base_dir, "DECA")
    data_dir = os.path.join(deca_dir, "data")
    
    # Step 1: Clone DECA from GitHub
    print("[1/4] Cloning DECA from GitHub...")
    if os.path.exists(deca_dir):
        print(f"DECA folder exists at {deca_dir}")
        response = input("Delete and re-clone? (y/n): ").strip().lower()
        if response == 'y':
            run_cmd(f"rm -rf {deca_dir}")
        else:
            print("Skipping clone...")
    
    if not os.path.exists(deca_dir):
        success = run_cmd("git clone https://github.com/YadiraF/DECA.git")
        if not success:
            print("[FAIL] Could not clone DECA. Check internet connection.")
            return 1
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Step 2: Download DECA model
    print()
    print("[2/4] Downloading DECA trained model...")
    deca_model_path = os.path.join(data_dir, "deca_model.tar")
    
    if os.path.exists(deca_model_path):
        print(f"[OK] deca_model.tar already exists")
    else:
        # DECA model is hosted on Google Drive
        # File ID: 1rp8UQQpx5D0RDdhB7Y8uQzb1H3pRbSjE (from DECA repo)
        print("DECA model needs to be downloaded from Google Drive")
        print()
        print("Option 1: Download manually and upload")
        print("  URL: https://drive.google.com/file/d/1rp8UQQpx5D0RDdhB7Y8uQzb1H3pRbSjE")
        print()
        print("Option 2: Use gdown (installing now...)")
        run_cmd("pip install gdown -q")
        
        try:
            import gdown
            gdown.download(
                "https://drive.google.com/uc?id=1rp8UQQpx5D0RDdhB7Y8uQzb1H3pRbSjE",
                deca_model_path,
                quiet=False
            )
            print("[OK] Downloaded deca_model.tar")
        except Exception as e:
            print(f"[WARN] gdown failed: {e}")
            print("Please download manually and upload to:")
            print(f"  {deca_model_path}")
    
    # Step 3: Download FLAME model
    print()
    print("[3/4] Downloading FLAME model...")
    flame_path = os.path.join(data_dir, "generic_model.pkl")
    
    if os.path.exists(flame_path):
        print(f"[OK] generic_model.pkl already exists")
    else:
        print()
        print("FLAME model requires registration at:")
        print("  https://flame.is.tue.mpg.de/")
        print()
        print("After registration, download 'FLAME 2020' and extract:")
        print("  - generic_model.pkl -> DECA/data/generic_model.pkl")
        print()
        print("For now, you can use the FLAME model from your local machine:")
        print("  d:\\Magicface_gravity\\DECA\\data\\generic_model.pkl")
    
    # Step 4: Download face landmark model
    print()
    print("[4/4] Downloading face landmark model...")
    landmark_path = os.path.join(data_dir, "landmark_embedding.npy")
    
    if os.path.exists(landmark_path):
        print(f"[OK] landmark_embedding.npy already exists")
    else:
        # This is usually included in the DECA repo or can be generated
        print("landmark_embedding.npy should be in DECA/data/")
        print("If missing, upload from local: d:\\Magicface_gravity\\DECA\\data\\")
    
    # Summary
    print()
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"DECA folder: {deca_dir}")
    print()
    print("Required files in DECA/data/:")
    
    required_files = [
        "deca_model.tar",
        "generic_model.pkl",
        "landmark_embedding.npy",
        "mean_texture.jpg",
        "texture_data_256.npy",
        "fixed_displacement_256.npy",
        "head_template.obj",
        "uv_face_eye_mask.png",
        "uv_face_mask.png",
    ]
    
    for f in required_files:
        fpath = os.path.join(data_dir, f)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  [OK] {f} ({size:.1f} MB)")
        else:
            print(f"  [MISSING] {f}")
    
    print()
    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Upload missing files from local d:\\Magicface_gravity\\DECA\\data\\")
    print("2. Run: python patch_chumpy.py")
    print("3. Test: python magicface_deca_pipeline.py test_images/image.jpg smile")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
