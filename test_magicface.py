"""
Enhanced Identity-Preserving Face Pipeline (Final Anti-Blur Version)

This version is optimized to produce sharp, high-quality results by:
- Using sharper interpolation methods.
- Reducing excessive blur on blending masks.
- Applying a final unsharp mask for crispness.
- Including debug outputs to diagnose issues.
"""

import os
import sys
import subprocess
import cv2
import torch
import numpy as np
import gc
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants - FIXED PATHS
ROOT_DIR = "/teamspace/studios/this_studio/Magicface"
MAGICFACE_DIR = os.path.join(ROOT_DIR, "MagicFace")
DECA_DIR = os.path.join(ROOT_DIR, "DECA")
CODEFORMER_DIR = os.path.join(ROOT_DIR, "models/CodeFormer")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
TEMP_DIR = os.path.join(ROOT_DIR, "temp")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Add DECA to path
sys.path.insert(0, DECA_DIR)

class EnhancedIdentityPreservingPipeline:
    """Pipeline focused on maximizing identity preservation and image sharpness."""
    
    def __init__(self, use_fp16=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = use_fp16 and self.device == "cuda"
        logger.info(f"Device: {self.device} | FP16: {self.use_fp16}")
        
        # Load MagicFace compatibility
        sys.path.insert(0, MAGICFACE_DIR)
        try:
            import compatibility
            logger.info("✓ Compatibility module loaded")
        except ImportError:
            pass
        
        self.app = None
        self.swapper = None
        self.deca = None
        self._load_face_models()
        self._load_deca()
    
    def _load_face_models(self):
        logger.info("Loading InsightFace models...")
        buffalo_root = os.path.join(ROOT_DIR, "Model/models")
        self.app = FaceAnalysis(name='buffalo_l', root=buffalo_root,
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.4)
        
        inswapper_path = os.path.join(MODELS_DIR, "inswapper_128.onnx")
        if os.path.exists(inswapper_path):
            self.swapper = insightface.model_zoo.get_model(inswapper_path)
            logger.info("✓ Inswapper loaded")
        else:
            logger.error(f"Inswapper not found at {inswapper_path}")

    def _load_deca(self):
        logger.info("Loading DECA model...")
        try:
            from decalib.deca import DECA
            from decalib.utils.config import cfg as deca_cfg
            
            deca_cfg.model.use_tex = False
            deca_cfg.rasterizer_type = 'pytorch3d' 

            # MONKEY PATCH: Skip renderer setup to avoid CUDA compilation error
            original_setup_renderer = DECA._setup_renderer
            def safe_setup_renderer(self, model_cfg):
                logger.warning("Skipping DECA renderer setup (Metrics-only mode)")
                pass
            
            DECA._setup_renderer = safe_setup_renderer
            
            try:
                self.deca = DECA(config=deca_cfg, device=self.device)
                logger.info("✓ DECA loaded successfully (Metrics-only mode)")
            finally:
                DECA._setup_renderer = original_setup_renderer

        except Exception as e:
            logger.error(f"Failed to load DECA: {e}")
            self.deca = None

    def save_debug_image(self, image, filename):
        """Saves an image to the output directory for debugging."""
        path = os.path.join(OUTPUT_DIR, f"debug_{filename}.png")
        cv2.imwrite(path, image)
        logger.info(f"Saved debug image: {path}")

    def run_deca_analysis(self, image):
        """Run DECA to get expression and shape parameters"""
        if self.deca is None:
            return None, None
        
        try:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().to(self.device) / 255.0
            img_t = img_t.unsqueeze(0)
            
            with torch.no_grad():
                codedict = self.deca.encode(img_t)
                return codedict['exp'].cpu().numpy(), codedict['shape'].cpu().numpy()
        except Exception as e:
            logger.warning(f"DECA analysis failed: {e}")
            return None, None

    def run_magicface(self, source_path, au_string, au_intensity):
        """Run MagicFace with error handling"""
        try:
            env = os.environ.copy()
            pythonpath = MAGICFACE_DIR
            if "PYTHONPATH" in env:
                pythonpath = MAGICFACE_DIR + os.pathsep + env["PYTHONPATH"]
            env["PYTHONPATH"] = pythonpath
            
            logger.info("[STAGE 1] MagicFace: Preprocessing...")
            crop_path = os.path.join(TEMP_DIR, "cropped.png")
            subprocess.run([
                "python", os.path.join(MAGICFACE_DIR, "utils", "preprocess.py"),
                "--img_path", source_path,
                "--save_path", crop_path
            ], cwd=MAGICFACE_DIR, env=env, check=True, capture_output=True)
            
            logger.info("[STAGE 1] MagicFace: Retrieving background...")
            bg_path = os.path.join(TEMP_DIR, "bg.png")
            subprocess.run([
                "python", os.path.join(MAGICFACE_DIR, "utils", "retrieve_bg.py"),
                "--img_path", crop_path,
                "--save_path", bg_path
            ], cwd=MAGICFACE_DIR, env=env, check=True, capture_output=True)
            
            magic_dir = os.path.join(TEMP_DIR, "magic_out")
            os.makedirs(magic_dir, exist_ok=True)
            
            logger.info("[STAGE 1] MagicFace: Generating expression...")
            subprocess.run([
                "python", "inference.py", 
                "--img_path", crop_path,
                "--bg_path", bg_path,
                "--au_test", au_string,
                "--AU_variation", au_intensity,
                "--saved_path", magic_dir,
                "--denoising_unet_path", os.path.join(MAGICFACE_DIR, "denoising_unet"),
                "--ID_unet_path", os.path.join(MAGICFACE_DIR, "ID_enc")
            ], cwd=MAGICFACE_DIR, env=env, check=True)
            
            logger.info("✓ MagicFace complete")
            return crop_path, os.path.join(magic_dir, os.path.basename(crop_path))
        except subprocess.CalledProcessError as e:
            logger.error(f"MagicFace failed: {e}")
            raise
    
    def identity_aware_blend(self, base_img, expr_img, base_face, expr_face, blend_strength=0.5):
        """
        ENHANCED BLENDING: Protects identity regions while applying expression.
        Uses sharper interpolation and reduced mask blur for a crisper result.
        """
        if base_img.shape != expr_img.shape:
            # ENHANCEMENT: Use sharper interpolation for resizing
            expr_img = cv2.resize(expr_img, (base_img.shape[1], base_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # ENHANCED ALIGNMENT: Use affine transform with 3 stable points
        if hasattr(base_face, 'landmark_2d_106') and hasattr(expr_face, 'landmark_2d_106'):
            base_lm = base_face.landmark_2d_106
            expr_lm = expr_face.landmark_2d_106
            
            base_lm = np.array(base_lm, dtype=np.float32)
            expr_lm = np.array(expr_lm, dtype=np.float32)
            
            key_indices = [33, 87, 8]  # Left eye corner, right eye corner, chin
            src_pts = expr_lm[key_indices]
            dst_pts = base_lm[key_indices]
            
            try:
                M = cv2.getAffineTransform(src_pts, dst_pts)
                # ENHANCEMENT: Use sharper interpolation for warping
                expr_img = cv2.warpAffine(expr_img, M, (base_img.shape[1], base_img.shape[0]), 
                                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
                logger.info("Applied enhanced affine warping for accurate alignment")
            except Exception as e:
                 logger.warning(f"Enhanced warping failed, falling back to original: {e}")

        # IDENTITY-AWARE MASKING
        h, w = base_img.shape[:2]
        identity_mask = np.zeros((h, w), dtype=np.float32)
        
        lm = np.array(base_face.landmark_2d_106, dtype=np.int32)
        
        # Eyes
        left_eye = lm[33:43]
        right_eye = lm[87:97]
        for eye_pts in [left_eye, right_eye]:
            if len(eye_pts) > 0:
                hull = cv2.convexHull(eye_pts)
                cv2.fillConvexPoly(identity_mask, hull, 1.0)
        
        # Nose
        nose_pts = lm[49:60]
        if len(nose_pts) > 0:
            hull = cv2.convexHull(nose_pts)
            cv2.fillConvexPoly(identity_mask, hull, 1.0)
        
        # Eyebrows
        left_brow = lm[19:26]
        right_brow = lm[27:34]
        for brow_pts in [left_brow, right_brow]:
            if len(brow_pts) > 0:
                hull = cv2.convexHull(brow_pts)
                cv2.fillConvexPoly(identity_mask, hull, 1.0)
    
        # ENHANCEMENT: Reduce mask blur for sharper edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        identity_mask = cv2.dilate(identity_mask, kernel, iterations=1)
        identity_mask = cv2.GaussianBlur(identity_mask, (7, 7), 2.5) # Reduced from (15,15), 5
        
        expression_mask = 1.0 - identity_mask
        
        # Create a base expression blend
        mouth_mask = np.zeros((h, w), dtype=np.float32)
        mouth_pts = lm[52:72]
        if len(mouth_pts) > 0:
            hull = cv2.convexHull(mouth_pts)
            cv2.fillConvexPoly(mouth_mask, hull, 1.0)
        # ENHANCEMENT: Reduce mask blur for sharper edges
        mouth_mask = cv2.GaussianBlur(mouth_mask, (25, 25), 12) # Reduced from (51,51), 25
        
        # Final Blending Logic
        mask_3ch = np.stack([mouth_mask, mouth_mask, mouth_mask], axis=2)
        identity_mask_3ch = np.stack([identity_mask, identity_mask, identity_mask], axis=2)
        
        base_preserved = base_img.astype(np.float32) * identity_mask_3ch
        expression_applied = (base_img.astype(np.float32) * (1 - mask_3ch * blend_strength) + \
                              expr_img.astype(np.float32) * mask_3ch * blend_strength) * (1 - identity_mask_3ch)
        
        final_blend = base_preserved + expression_applied
        return np.clip(final_blend, 0, 255).astype(np.uint8)

    def run_face_swap(self, source_path, target_img):
        """Face swap with multi-face handling"""
        try:
            source_img_cv = cv2.imread(source_path)
            if source_img_cv is None:
                logger.error(f"Could not read source: {source_path}")
                return target_img, 0.0

            source_faces = self.app.get(source_img_cv)
            if not source_faces:
                logger.warning("No face in source")
                return target_img, 0.0
            
            source_face = max(source_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            source_emb = source_face.embedding.reshape(1, -1)
            
            target_faces = self.app.get(target_img)
            if not target_faces:
                logger.warning("No face in target")
                return target_img, 0.0
            
            target_face = max(target_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            
            result = self.swapper.get(target_img, target_face, source_face, paste_back=True)
            
            result_faces = self.app.get(result)
            if result_faces:
                result_face = max(result_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                result_emb = result_face.embedding.reshape(1, -1)
                similarity = cosine_similarity(source_emb, result_emb)[0][0]
                return result, similarity
            return result, 0.0
        except Exception as e:
            logger.error(f"Face swap failed: {e}")
            import traceback
            traceback.print_exc()
            return target_img, 0.0
    
    def run_codeformer(self, image, fidelity=0.9):
        """
        Face enhancement using CodeFormer.
        TUNING: Lower fidelity (e.g., 0.85) can add detail but may reduce identity.
        Higher fidelity (e.g., 0.95) preserves identity but may be less sharp.
        """
        logger.info(f"[STAGE 5] CodeFormer Quality Enhancement (fidelity: {fidelity})")
        
        try:
            sys.path.insert(0, CODEFORMER_DIR)
            from basicsr.archs.codeformer_arch import CodeFormer
            from torchvision.transforms.functional import normalize
            
            net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                             connect_list=['32', '64', '128', '256']).to(self.device)
            
            ckpt_path = os.path.join(CODEFORMER_DIR, "weights/CodeFormer/codeformer.pth")
            if not os.path.exists(ckpt_path):
                logger.error(f"✗ CodeFormer model not found at {ckpt_path}")
                return image
            
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            net.load_state_dict(checkpoint['params_ema'])
            net.eval()
            
            h, w = image.shape[:2]
            img_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            face_t = torch.from_numpy(img).permute(2, 0, 1).float().to(self.device).unsqueeze(0)
            normalize(face_t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            
            with torch.no_grad():
                output = net(face_t, w=fidelity, adain=True)[0]
                output = output.squeeze(0).clamp(-1, 1)
                output = (output + 1) / 2
                output = output.permute(1, 2, 0).cpu().numpy()
                output = (output * 255).astype(np.uint8)
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            # Use sharper interpolation when resizing back
            output = cv2.resize(output, (w, h), interpolation=cv2.INTER_CUBIC)
            logger.info("✓ CodeFormer enhancement complete")
            del net
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return output
            
        except Exception as e:
            logger.warning(f"CodeFormer failed: {e}, using original image")
            return image

    def apply_unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, amount=1.5):
        """Applies an unsharp mask to sharpen an image."""
        logger.info("Applying unsharp mask for final sharpening.")
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def enhance_identity(self, source_img, result_img, similarity_threshold=0.96):
        """Final targeted enhancement to boost identity score without disrupting expression."""
        logger.info(f"[STAGE 6] Final Identity Enhancement (Target: {similarity_threshold})")
        
        source_faces = self.app.get(source_img)
        result_faces = self.app.get(result_img)
        
        if not source_faces or not result_faces:
            return result_img
        
        source_face = max(source_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        result_face = max(result_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        
        source_emb = source_face.embedding.reshape(1, -1)
        result_emb = result_face.embedding.reshape(1, -1)
        similarity = cosine_similarity(source_emb, result_emb)[0][0]
        
        logger.info(f"  Current identity similarity: {similarity:.4f}")
        
        if similarity < similarity_threshold:
            temp_source_path = os.path.join(TEMP_DIR, "temp_source_for_enhance.jpg")
            cv2.imwrite(temp_source_path, source_img)
            
            enhanced_result, new_similarity = self.run_face_swap(temp_source_path, result_img)
            
            if new_similarity > similarity:
                logger.info(f"  ✓ Identity enhanced: {similarity:.4f} -> {new_similarity:.4f}")
                return enhanced_result
            else:
                logger.info(f"  - Enhancement did not improve score. Keeping previous result.")
        
        return result_img

    def process(self, source_path, au_string="AU26", au_intensity="4", 
                blend_strength=0.5,
                use_codeformer=True,
                codeformer_fidelity=0.9, # Default tuned for a balance of sharpness and identity
                refinement_passes=5,
                identity_threshold=0.96,
                smart_intensity=True,
                max_intensity=7,
                exp_diff_threshold=1.0):
        """
        Enhanced main pipeline with a focus on identity preservation and sharpness.
        """
        logger.info("ENHANCED IDENTITY-PRESERVING PIPELINE - START")
        logger.info(f"Source: {os.path.basename(source_path)} | AU: {au_string} (Init Intensity: {au_intensity})")
        
        try:
            source_img_cv = cv2.imread(source_path)
            start_exp, start_shape = self.run_deca_analysis(source_img_cv)
            
            # --- SMART INTENSITY LOOP ---
            current_intensity = int(au_intensity)
            best_expression_img = None
            best_exp_diff = 0.0
            loop_range = range(current_intensity, max_intensity + 1) if smart_intensity else [current_intensity]
            
            for intensity in loop_range:
                logger.info(f"\n[Generation Loop] Trying Intensity: {intensity}")
                cropped_path, magic_result_path = self.run_magicface(source_path, au_string, str(intensity))
                expression_img = cv2.imread(magic_result_path)
                
                self.save_debug_image(expression_img, f"01_magicface_intensity_{intensity}")

                curr_exp, _ = self.run_deca_analysis(expression_img)
                exp_diff = np.linalg.norm(start_exp - curr_exp) if (start_exp is not None and curr_exp is not None) else 0.0
                logger.info(f"  Raw Expression Strength (DECA L2): {exp_diff:.4f}")
                
                if exp_diff >= exp_diff_threshold:
                    logger.info("  ✓ Expression strength is sufficient.")
                    best_expression_img = expression_img
                    best_exp_diff = exp_diff
                    break
                else:
                    logger.info("  ⚠ Expression too weak. Increasing intensity...")
                    best_expression_img = expression_img
                    best_exp_diff = exp_diff
            
            logger.info("----------------------------------------")
            logger.info(f"Selected Expression Image (Strength: {best_exp_diff:.4f})")
            expression_img = best_expression_img
            
            orig_faces = self.app.get(source_img_cv)
            expr_faces = self.app.get(expression_img)
            orig_face = max(orig_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) if orig_faces else None
            expr_face = max(expr_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) if expr_faces else None

            # Stage 2: Initial Face Swap
            logger.info("[STAGE 2] Initial Face Swap (Identity Restoration)")
            current_result, _ = self.run_face_swap(source_path, expression_img)
            self.save_debug_image(current_result, "02_after_initial_swap")
            
            # Stage 3: Identity-Aware Blending
            logger.info("[STAGE 3] Identity-Aware Expression Blending")
            swap_faces = self.app.get(current_result)
            if swap_faces and orig_face and expr_face:
                swap_face = max(swap_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                current_result = self.identity_aware_blend(current_result, expression_img, swap_face, expr_face, blend_strength)
            self.save_debug_image(current_result, "03_after_blending")

            # Stage 4: Iterative Refinement Loop
            logger.info(f"[STAGE 4] Iterative Identity Refinement (Max passes: {refinement_passes})")
            best_result = current_result.copy()
            best_id_sim = 0.0
            _, current_sim = self.run_face_swap(source_path, current_result)
            
            for i in range(refinement_passes):
                logger.info(f"--- Refinement Pass {i+1} ---")
                new_result, new_sim = self.run_face_swap(source_path, current_result)
                improvement = new_sim - current_sim
                logger.info(f"Identity: {current_sim:.4f} -> {new_sim:.4f} (Improvement: {improvement:.4f})")
                
                if new_sim > best_id_sim:
                    best_id_sim = new_sim
                    best_result = new_result.copy()
                
                current_result = new_result
                current_sim = new_sim
                
                if improvement < 0.001 or new_sim >= identity_threshold:
                    logger.info("Stopping refinement (plateau or threshold reached).")
                    break

            final_result = best_result
            final_sim = best_id_sim
            self.save_debug_image(final_result, "04_after_refinement")
            
            # Stage 5: CodeFormer Clarity
            if use_codeformer:
                final_result = self.run_codeformer(final_result, fidelity=codeformer_fidelity)
                self.save_debug_image(final_result, "05_after_codeformer")

            # Stage 6: Final Identity Enhancement
            final_result = self.enhance_identity(source_img_cv, final_result, identity_threshold)
            self.save_debug_image(final_result, "06_after_identity_enhance")

            # Stage 7: Final Sharpening
            final_result = self.apply_unsharp_mask(final_result, kernel_size=(5,5), sigma=1.0, amount=1.2)
            
            # Final Verification
            logger.info("[STAGE 8] Final Verification")
            final_exp, _ = self.run_deca_analysis(final_result)
            exp_diff = np.linalg.norm(start_exp - final_exp) if (start_exp is not None and final_exp is not None) else 0.0
            
            final_faces = self.app.get(final_result)
            if final_faces:
                final_face_obj = max(final_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                final_emb = final_face_obj.embedding.reshape(1, -1)
                source_emb = orig_face.embedding.reshape(1, -1)
                final_sim = cosine_similarity(source_emb, final_emb)[0][0]

            final_path = os.path.join(OUTPUT_DIR, "final_result_sad.png")
            cv2.imwrite(final_path, final_result)
            
            logger.info(f"Final Identity: {final_sim*100:.2f}% | Exp Diff: {exp_diff:.4f}")
            logger.info("PIPELINE COMPLETE")
            return final_path, final_sim, exp_diff
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0, 0.0


if __name__ == "__main__":
    pipeline = EnhancedIdentityPreservingPipeline(use_fp16=True)
    
    # UPDATE THIS to your image path
    source = "/teamspace/studios/this_studio/Magicface/MagicFace/test_images/ros1.jpg"
    
    final_output, similarity, exp_diff = pipeline.process(
        source_path=source,
        # --- CHANGE FOR SAD EXPRESSION ---
        au_string="AU1+AU4+AU15+AU17",  # AUs for a sad expression
        au_intensity="4",               # Moderate intensity for sadness
        # --- END OF CHANGE ---
        blend_strength=0.5,
        use_codeformer=True,
        codeformer_fidelity=0.9,
        smart_intensity=True,
        exp_diff_threshold=1.0  # Lowered threshold for a more subtle expression
    )
    
    if final_output:
        print(f"\nSUCCESS! Output saved at: {final_output}")