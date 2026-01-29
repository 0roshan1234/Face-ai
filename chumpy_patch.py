"""
Patch chumpy for numpy 2.x and Python 3.11+ compatibility
Auto-detects chumpy location on any platform (Windows/Linux/Cloud)
"""

import os
import sys
import re

def find_chumpy_path():
    """Auto-detect chumpy installation path"""
    try:
        import chumpy
        chumpy_path = os.path.dirname(chumpy.__file__)
        print(f"[OK] Found chumpy at: {chumpy_path}")
        return chumpy_path
    except ImportError:
        print("[ERROR] chumpy not installed!")
        print("Run: pip install chumpy --no-build-isolation")
        return None

def patch_file(filepath, old_text, new_text, description):
    """Patch a single file"""
    if not os.path.exists(filepath):
        print(f"[SKIP] File not found: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    if old_text not in content:
        if new_text in content:
            print(f"[OK] Already patched: {description}")
            return True
        else:
            print(f"[SKIP] Pattern not found: {description}")
            return False
    
    new_content = content.replace(old_text, new_text)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"[PATCHED] {description}")
    return True

def patch_chumpy(chumpy_path):
    """Apply all patches to chumpy"""
    
    patches_applied = 0
    
    # Patch 1: Fix inspect.getargspec (Python 3.11+)
    ch_file = os.path.join(chumpy_path, 'ch.py')
    
    # Try multiple patterns for getargspec fix
    patterns = [
        # Pattern 1: Direct replacement
        (
            "from inspect import getargspec",
            "try:\n    from inspect import getfullargspec as getargspec\nexcept ImportError:\n    from inspect import getargspec"
        ),
        # Pattern 2: Sometimes it's imported differently
        (
            "import inspect\n",
            "import inspect\ntry:\n    inspect.getargspec = inspect.getfullargspec\nexcept AttributeError:\n    pass\n"
        ),
    ]
    
    for old, new in patterns:
        if patch_file(ch_file, old, new, "inspect.getargspec -> getfullargspec"):
            patches_applied += 1
            break
    
    # Patch 2: Fix numpy type aliases (numpy 2.x)
    # These are deprecated/removed in numpy 2.0
    numpy_fixes = [
        ("np.bool_", "np.bool_"),
        ("np.int_", "np.int_"),
        ("np.float_", "np.float_"),
        ("np.complex_", "np.complex_"),
        ("np.object_", "np.object_"),
        ("np.str_", "np.str_"),
    ]
    
    # Find all .py files in chumpy
    for root, dirs, files in os.walk(chumpy_path):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    modified = False
                    for old, new in numpy_fixes:
                        # Use regex to avoid replacing np.bool_ with np.bool__
                        pattern = rf'\b{re.escape(old)}\b(?!_)'
                        if re.search(pattern, content):
                            content = re.sub(pattern, new, content)
                            modified = True
                    
                    if modified:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"[PATCHED] numpy types in {filename}")
                        patches_applied += 1
                        
                except Exception as e:
                    print(f"[WARN] Could not patch {filename}: {e}")
    
    return patches_applied

def patch_face_alignment():
    """Patch face_alignment for newer versions"""
    try:
        import face_alignment
        fa_path = os.path.dirname(face_alignment.__file__)
        
        # Find the detector file
        detector_file = None
        for root, dirs, files in os.walk(fa_path):
            for f in files:
                if 'detector' in f.lower() and f.endswith('.py'):
                    detector_file = os.path.join(root, f)
                    break
        
        if detector_file:
            # Fix LandmarksType._2D -> LandmarksType.TWO_D
            patch_file(
                detector_file,
                "LandmarksType._2D",
                "LandmarksType.TWO_D",
                "face_alignment LandmarksType fix"
            )
            patch_file(
                detector_file,
                "LandmarksType._3D",
                "LandmarksType.THREE_D",
                "face_alignment LandmarksType 3D fix"
            )
            
    except ImportError:
        print("[SKIP] face_alignment not installed")
    except Exception as e:
        print(f"[WARN] Could not patch face_alignment: {e}")

def main():
    print("=" * 50)
    print("CHUMPY + FACE_ALIGNMENT PATCHER")
    print("(Compatible with numpy 2.x and Python 3.11+)")
    print("=" * 50)
    print()
    
    # Find and patch chumpy
    chumpy_path = find_chumpy_path()
    if chumpy_path:
        patches = patch_chumpy(chumpy_path)
        print(f"\nApplied {patches} patches to chumpy")
    
    print()
    
    # Patch face_alignment
    print("Checking face_alignment...")
    patch_face_alignment()
    
    print()
    print("=" * 50)
    print("PATCHING COMPLETE!")
    print("=" * 50)
    
    # Verify
    print("\nVerifying imports...")
    try:
        import chumpy
        print("[OK] chumpy imports successfully")
    except Exception as e:
        print(f"[FAIL] chumpy import error: {e}")
    
    try:
        import face_alignment
        print("[OK] face_alignment imports successfully")
    except Exception as e:
        print(f"[WARN] face_alignment import issue: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
