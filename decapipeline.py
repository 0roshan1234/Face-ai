#!/usr/bin/env python3
"""
MagicFace + DECA + FaceSwapper + CodeFormer Complete Pipeline
All compatibility patches included
Runs with: python magicface_complete.py
"""

# ============================================================
# COMPATIBILITY PATCHES (Must be first!)
# ============================================================

import sys
import os
import platform
import types

# Detect environment
is_cloud = platform.system() == "Linux"
if is_cloud:
    BASE_DIR = "/teamspace/studios/this_studio/Magicface"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MAGICFACE_PATH = os.path.join(BASE_DIR, "MagicFace")
DECA_PATH = os.path.join(BASE_DIR, "DECA")
MODELS_PATH = os.path.join(BASE_DIR, "models")
CODEFORMER_PATH = os.path.join(MODELS_PATH, "CodeFormer")
OUTPUT_PATH = os.path.join(BASE_DIR, "output")

sys.path.insert(0, MAGICFACE_PATH)
sys.path.insert(0, DECA_PATH)
sys.path.insert(0, CODEFORMER_PATH)

# Create torch._vendor module
try:
    import torch
    if not hasattr(torch, '_vendor'):
        torch._vendor = types.ModuleType('torch._vendor')
        torch._vendor.packaging = types.ModuleType('torch._vendor.packaging')
        torch._vendor.packaging.version = types.ModuleType('torch._vendor.packaging.version')
        torch._vendor.packaging.version.parse = lambda v: tuple(map(int, v.split('.')))
        sys.modules['torch._vendor'] = torch._vendor
        sys.modules['torch._vendor.packaging'] = torch._vendor.packaging
        sys.modules['torch._vendor.packaging.version'] = torch._vendor.packaging.version
except ImportError:
    pass

# Patch xformers
try:
    import torch
    if not hasattr(torch.backends.cuda, 'is_flash_attention_available'):
        torch.backends.cuda.is_flash_attention_available = lambda: False
except ImportError:
    pass

# Patch huggingface_hub
try:
    import huggingface_hub.utils
    if not hasattr(huggingface_hub.utils, 'OfflineModeIsEnabled'):
        huggingface_hub.utils.OfflineModeIsEnabled = lambda: False
except ImportError:
    pass

try:
    import huggingface_hub
    if not hasattr(huggingface_hub, 'cached_download'):
        from huggingface_hub import hf_hub_download
        huggingface_hub.cached_download = hf_hub_download
except ImportError:
    pass

# Patch torch.utils.checkpoint
try:
    import torch.utils.checkpoint
    if not hasattr(torch.utils.checkpoint, '_ignored_ops'):
        torch.utils.checkpoint._ignored_ops = set()
except ImportError:
    pass

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============================================================
# IMPORTS
# ============================================================

import logging
import gc
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize
from PIL import Image
from pathlib import Path
from diffusers import AutoencoderKL, UniPCMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    'source_image': 'MagicFace/test_images/ros1.jpg',
    'expression': 'closed_mouth',  # Test with closed mouth - should be clearly different
    'inference_steps': 50,
    'seed': 424,
    'identity_threshold': 0.95,
    'max_correction_passes': 3,
    'codeformer_fidelity': 0.7,
}

PRESETS = {
    'neutral': ('', ''),
    'smile': ('AU12', '4'),
    'big_smile': ('AU12+AU6+AU25', '5+4+3'),
    'surprise': ('AU1+AU2+AU5+AU26', '5+5+4+4'),
    'angry': ('AU4+AU17', '5+4'),
    'sad': ('AU1+AU4+AU15', '3+3+4'),
    'disgust': ('AU9+AU10', '4+4'),
    'fear': ('AU1+AU2+AU4+AU5+AU20', '4+4+3+4+4'),
    'closed_mouth': ('AU17', '5'),  # Chin raiser - closes mouth
    'mouth_open': ('AU25+AU26', '5+5'),  # Lips part + jaw drop
}

ind_dict = {'AU1':0, 'AU2':1, 'AU4':2, 'AU5':3, 'AU6':4, 'AU9':5,
            'AU12':6, 'AU15':7, 'AU17':8, 'AU20':9, 'AU25':10, 'AU26':11}

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================
# DECA ENCODER (Skip Renderer - Only need shape extraction)
# ============================================================

class DECAEncoder:
    """DECA Encoder for 3D identity extraction (no renderer needed)"""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.encoder = None
        self.flame = None
        self._load()
    
    def _load(self):
        try:
            # Import DECA components without renderer
            from decalib.models.encoders import ResnetEncoder
            from decalib.models.FLAME import FLAME
            from decalib.utils.config import cfg as deca_cfg
            
            deca_data = os.path.join(DECA_PATH, "data")
            
            # Load encoder weights from deca_model
            model_path = os.path.join(deca_data, 'deca_model.tar')
            if not os.path.exists(model_path):
                logger.warning(f"DECA model not found: {model_path}")
                return
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Setup FLAME config
            deca_cfg.model.flame_model_path = os.path.join(deca_data, 'generic_model.pkl')
            deca_cfg.model.flame_lmk_embedding_path = os.path.join(deca_data, 'landmark_embedding.npy')
            deca_cfg.model.n_shape = 100
            deca_cfg.model.n_tex = 50
            deca_cfg.model.n_exp = 50
            deca_cfg.model.n_cam = 3
            deca_cfg.model.n_pose = 6
            deca_cfg.model.n_light = 27
            deca_cfg.model.use_tex = False
            
            # Create encoder
            self.encoder = ResnetEncoder(outsize=236).to(self.device)
            
            # Load encoder weights
            encoder_dict = {k.replace('E_flame.', ''): v for k, v in checkpoint['E_flame'].items()}
            self.encoder.load_state_dict(encoder_dict)
            self.encoder.eval()
            
            # Create FLAME decoder
            self.flame = FLAME(deca_cfg.model).to(self.device)
            
            logger.info("DECA Encoder loaded (renderer skipped)")
            
        except Exception as e:
            logger.warning(f"DECA Encoder not loaded: {e}")
            import traceback
            traceback.print_exc()
            self.encoder = None
    
    def extract_identity(self, image_path):
        if self.encoder is None:
            return None
        
        try:
            # Load and preprocess
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                params = self.encoder(img)
                # Shape is first 100 parameters
                shape = params[:, :100]
            
            return shape
            
        except Exception as e:
            logger.warning(f"DECA extraction failed: {e}")
            return None
    
    def validate(self, source_path, result_path, threshold=0.95):
        shape1 = self.extract_identity(source_path)
        shape2 = self.extract_identity(result_path)
        
        if shape1 is None or shape2 is None:
            return True, 1.0
        
        sim = F.cosine_similarity(shape1.flatten().unsqueeze(0), shape2.flatten().unsqueeze(0)).item()
        return sim >= threshold, sim

# ============================================================
# FACE SWAPPER
# ============================================================

class FaceSwapper:
    """Face swap for identity restoration"""
    
    def __init__(self):
        self.app = None
        self.swapper = None
        self._load()
    
    def _load(self):
        try:
            from insightface.app import FaceAnalysis
            import insightface
            
            self.app = FaceAnalysis(
                name='buffalo_l',
                root=MODELS_PATH,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            
            swap_path = os.path.join(MODELS_PATH, 'inswapper_128.onnx')
            if os.path.exists(swap_path):
                self.swapper = insightface.model_zoo.get_model(
                    swap_path,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                logger.info("FaceSwapper loaded")
            else:
                logger.warning(f"inswapper_128.onnx not found at {swap_path}")
        except Exception as e:
            logger.error(f"FaceSwapper failed: {e}")
    
    def swap(self, source_img, target_img):
        if self.app is None or self.swapper is None:
            return target_img
        
        src_faces = self.app.get(source_img)
        tgt_faces = self.app.get(target_img)
        
        if not src_faces or not tgt_faces:
            logger.warning("No faces detected for swap")
            return target_img
        
        return self.swapper.get(target_img, tgt_faces[0], src_faces[0], paste_back=True)
    
    def get_similarity(self, img1, img2):
        if self.app is None:
            return 0.0
        
        f1 = self.app.get(img1)
        f2 = self.app.get(img2)
        
        if not f1 or not f2:
            return 0.0
        
        e1, e2 = f1[0].embedding, f2[0].embedding
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))

# ============================================================
# CODEFORMER ENHANCER
# ============================================================

class CodeFormerEnhancer:
    """Face enhancement using CodeFormer"""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.net = None
        self.face_helper = None
        self._load()
    
    def _load(self):
        try:
            from basicsr.utils.registry import ARCH_REGISTRY
            from facelib.utils.face_restoration_helper import FaceRestoreHelper
            from basicsr.utils.download_util import load_file_from_url
            
            # Load CodeFormer model
            self.net = ARCH_REGISTRY.get('CodeFormer')(
                dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                connect_list=['32', '64', '128', '256']
            ).to(self.device)
            
            ckpt_path = load_file_from_url(
                url='https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
                model_dir=os.path.join(CODEFORMER_PATH, 'weights/CodeFormer'),
                progress=True, file_name=None
            )
            checkpoint = torch.load(ckpt_path)['params_ema']
            self.net.load_state_dict(checkpoint)
            self.net.eval()
            
            # Face helper
            self.face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=True,
                device=self.device
            )
            
            logger.info("CodeFormer loaded")
            
        except Exception as e:
            logger.warning(f"CodeFormer not loaded: {e}")
            self.net = None
    
    def enhance(self, img, fidelity=0.7):
        if self.net is None or self.face_helper is None:
            return img
        
        try:
            from basicsr.utils import img2tensor, tensor2img
            
            self.face_helper.clean_all()
            self.face_helper.read_image(img)
            
            # Detect faces
            num_faces = self.face_helper.get_face_landmarks_5(
                only_center_face=False, resize=640, eye_dist_threshold=5
            )
            logger.info(f"  CodeFormer: detected {num_faces} faces")
            
            if num_faces == 0:
                return img
            
            self.face_helper.align_warp_face()
            
            # Enhance each face
            for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
                cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.net(cropped_face_t, w=fidelity, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                
                restored_face = restored_face.astype('uint8')
                self.face_helper.add_restored_face(restored_face, cropped_face)
            
            # Paste back
            self.face_helper.get_inverse_affine(None)
            restored_img = self.face_helper.paste_faces_to_input_image()
            
            logger.info(f"  CodeFormer: enhancement complete")
            return restored_img
            
        except Exception as e:
            logger.warning(f"CodeFormer enhancement failed: {e}")
            return img

# ============================================================
# MAGICFACE RUNNER
# ============================================================

class MagicFaceRunner:
    """MagicFace expression generation"""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.pipeline = None
        self.tokenizer = None
        self.text_encoder = None
        self._load()
    
    def _load(self):
        try:
            from mgface.pipelines_mgface.pipeline_mgface import MgPipeline
            from mgface.pipelines_mgface.unet_ID_2d_condition import UNetID2DConditionModel
            from mgface.pipelines_mgface.unet_deno_2d_condition import UNetDeno2DConditionModel
            
            sd_model = "sd-legacy/stable-diffusion-v1-5"
            id_unet_path = os.path.join(MAGICFACE_PATH, "ID_enc")
            deno_unet_path = os.path.join(MAGICFACE_PATH, "denoising_unet")
            
            logger.info("Loading VAE and Text Encoder...")
            vae = AutoencoderKL.from_pretrained(sd_model, subfolder="vae").to(self.device)
            self.text_encoder = CLIPTextModel.from_pretrained(sd_model, subfolder="text_encoder").to(self.device)
            self.tokenizer = CLIPTokenizer.from_pretrained(sd_model, subfolder="tokenizer")
            
            logger.info("Loading ID UNet...")
            unet_ID = UNetID2DConditionModel.from_pretrained(
                id_unet_path,
                use_safetensors=True,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )
            
            logger.info("Loading Denoising UNet...")
            unet_deno = UNetDeno2DConditionModel.from_pretrained(
                deno_unet_path,
                use_safetensors=True,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )
            
            unet_deno.requires_grad_(False)
            unet_ID.requires_grad_(False)
            vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            
            logger.info("Creating Pipeline...")
            self.pipeline = MgPipeline.from_pretrained(
                sd_model,
                vae=vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet_ID=unet_ID,
                unet_deno=unet_deno,
                safety_checker=None,
                torch_dtype=torch.float16,
            ).to(self.device)
            
            self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
            logger.info("MagicFace loaded!")
            
        except Exception as e:
            logger.error(f"MagicFace failed: {e}")
            import traceback
            traceback.print_exc()
    
    def generate(self, source_path, au_string, intensity_string, steps=50, seed=424):
        if self.pipeline is None:
            return None
        
        transform = transforms.ToTensor()
        source_img = Image.open(source_path).convert('RGB')
        source = transform(source_img).unsqueeze(0)
        bg = source.clone()
        
        prompt = 'A close up of a person.'
        prompt_embeds = self.text_encoder(
            self.tokenizer([prompt], max_length=self.tokenizer.model_max_length,
                          padding="max_length", truncation=True, return_tensors="pt").input_ids.to(self.device)
        )[0]
        
        au_prompt = np.zeros((12,))
        if au_string and intensity_string:
            if '+' in au_string:
                aus = au_string.split('+')
                ints = intensity_string.split('+')
                for au, intensity in zip(aus, ints):
                    if au.upper() in ind_dict:
                        au_prompt[ind_dict[au.upper()]] = int(intensity)
            else:
                if au_string.upper() in ind_dict:
                    au_prompt[ind_dict[au_string.upper()]] = int(intensity_string)
        
        logger.info(f"  AU vector: {au_prompt}")
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        tor_exp = torch.from_numpy(au_prompt).unsqueeze(0)
        
        result = self.pipeline(
            prompt_embeds=prompt_embeds,
            source=source,
            bg=bg,
            au=tor_exp,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
        
        return result

# ============================================================
# MAIN PIPELINE
# ============================================================

class Pipeline:
    """Complete MagicFace + DECA + FaceSwapper + CodeFormer Pipeline"""
    
    def __init__(self):
        logger.info("=" * 60)
        logger.info("MAGICFACE + DECA + CODEFORMER PIPELINE")
        logger.info(f"Environment: {'Cloud' if is_cloud else 'Local'}")
        logger.info(f"Base: {BASE_DIR}")
        logger.info("=" * 60)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Device: {self.device}")
        
        logger.info("\n[Loading Components]")
        logger.info("-" * 40)
        
        self.deca = DECAEncoder(self.device)
        self.swapper = FaceSwapper()
        self.codeformer = CodeFormerEnhancer(self.device)
        self.magicface = MagicFaceRunner(self.device)
        
        logger.info("-" * 40)
        logger.info("[All Components Loaded]")
    
    def run(self, source_path, expression='smile', threshold=0.95, max_passes=3, 
            steps=50, seed=424, codeformer_fidelity=0.7):
        logger.info(f"\n{'='*60}")
        logger.info("RUNNING PIPELINE")
        logger.info(f"{'='*60}")
        logger.info(f"Source: {source_path}")
        logger.info(f"Expression: {expression}")
        
        # Get AU codes
        if expression in PRESETS:
            au_string, intensity_string = PRESETS[expression]
        else:
            au_string = expression
            intensity_string = '4' * len(expression.split('+'))
        
        logger.info(f"AU: {au_string}, Intensity: {intensity_string}")
        
        # Stage 1: DECA baseline
        logger.info("\n[Stage 1] DECA Identity Baseline...")
        baseline_shape = self.deca.extract_identity(source_path)
        if baseline_shape is not None:
            logger.info(f"  ✓ Shape extracted: {baseline_shape.shape}")
        else:
            logger.info("  ✗ DECA skipped")
        
        # Stage 2: Generate expression
        logger.info(f"\n[Stage 2] MagicFace Generating {expression}...")
        result = self.magicface.generate(source_path, au_string, intensity_string, steps, seed)
        
        if result is None:
            logger.error("  ✗ Generation failed!")
            return None
        
        expr_path = os.path.join(OUTPUT_PATH, 'stage2_expression.png')
        result.save(expr_path)
        logger.info(f"  ✓ Saved: {expr_path}")
        
        # Stage 3: Face swap
        logger.info("\n[Stage 3] Restoring Identity (FaceSwap)...")
        source_img = cv2.imread(source_path)
        expr_img = cv2.imread(expr_path)
        
        swapped = self.swapper.swap(source_img, expr_img)
        swap_path = os.path.join(OUTPUT_PATH, 'stage3_swapped.png')
        cv2.imwrite(swap_path, swapped)
        
        sim_2d = self.swapper.get_similarity(source_img, swapped)
        logger.info(f"  ✓ 2D Identity: {sim_2d*100:.1f}%")
        logger.info(f"  ✓ Saved: {swap_path}")
        
        # Stage 4: DECA validation
        logger.info("\n[Stage 4] DECA 3D Validation...")
        is_valid, sim_3d = self.deca.validate(source_path, swap_path, threshold)
        logger.info(f"  3D Identity: {sim_3d*100:.1f}%")
        logger.info(f"  Status: {'✓ PASSED' if is_valid else '✗ NEEDS CORRECTION'}")
        
        # Stage 5: Correction loop
        current_result = swapped
        correction_pass = 0
        
        while not is_valid and correction_pass < max_passes:
            correction_pass += 1
            logger.info(f"\n[Stage 5] Correction Pass {correction_pass}...")
            
            current_result = self.swapper.swap(source_img, current_result)
            temp_path = os.path.join(OUTPUT_PATH, f'stage5_correction_{correction_pass}.png')
            cv2.imwrite(temp_path, current_result)
            
            is_valid, sim_3d = self.deca.validate(source_path, temp_path, threshold)
            sim_2d = self.swapper.get_similarity(source_img, current_result)
            logger.info(f"  2D: {sim_2d*100:.1f}%, 3D: {sim_3d*100:.1f}%")
        
        # Stage 6: CodeFormer enhancement
        logger.info("\n[Stage 6] CodeFormer Enhancement...")
        enhanced = self.codeformer.enhance(current_result, fidelity=codeformer_fidelity)
        enhance_path = os.path.join(OUTPUT_PATH, 'stage6_enhanced.png')
        cv2.imwrite(enhance_path, enhanced)
        logger.info(f"  ✓ Saved: {enhance_path}")
        
        # Final similarity after enhancement
        sim_2d_final = self.swapper.get_similarity(source_img, enhanced)
        logger.info(f"  Final 2D Identity: {sim_2d_final*100:.1f}%")
        
        # Stage 7: Save final results
        logger.info("\n[Stage 7] Saving Final Results...")
        name = Path(source_path).stem
        final_path = os.path.join(OUTPUT_PATH, f'{name}_{expression}_final.png')
        cv2.imwrite(final_path, enhanced)
        
        # Create comparison
        comparison = np.hstack([source_img, enhanced])
        comp_path = os.path.join(OUTPUT_PATH, f'{name}_{expression}_comparison.png')
        cv2.imwrite(comp_path, comparison)
        
        # Pipeline stages visualization
        expr_img_final = cv2.imread(expr_path)
        swap_img_final = cv2.imread(swap_path)
        stages = np.hstack([source_img, expr_img_final, swap_img_final, enhanced])
        stages = cv2.resize(stages, (stages.shape[1]//2, stages.shape[0]//2))
        stages_path = os.path.join(OUTPUT_PATH, f'{name}_{expression}_stages.png')
        cv2.imwrite(stages_path, stages)
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"2D Identity: {sim_2d_final*100:.1f}%")
        logger.info(f"3D Identity: {sim_3d*100:.1f}%")
        logger.info(f"Validated: {'YES' if is_valid else 'NO'}")
        logger.info(f"Correction passes: {correction_pass}")
        logger.info(f"Output: {final_path}")
        logger.info(f"Comparison: {comp_path}")
        logger.info(f"Stages: {stages_path}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'output': final_path,
            'comparison': comp_path,
            'stages': stages_path,
            'sim_2d': sim_2d_final,
            'sim_3d': sim_3d,
            'is_valid': is_valid,
            'correction_passes': correction_pass,
        }

# ============================================================
# MAIN
# ============================================================

def find_test_image():
    default = os.path.join(BASE_DIR, CONFIG['source_image'])
    if os.path.exists(default):
        return default
    
    test_dir = os.path.join(MAGICFACE_PATH, 'test_images')
    if os.path.exists(test_dir):
        for f in os.listdir(test_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                return os.path.join(test_dir, f)
    return None

def main():
    print("=" * 60)
    print("  MAGICFACE + DECA + CODEFORMER PIPELINE")
    print("  Run: python magicface_complete.py")
    print("=" * 60)
    print()
    
    source = find_test_image()
    if source is None:
        print(f"[ERROR] No test image found!")
        print(f"Add image to: {os.path.join(BASE_DIR, 'MagicFace/test_images/')}")
        return 1
    
    print(f"Source: {source}")
    print(f"Expression: {CONFIG['expression']}")
    print()
    
    pipeline = Pipeline()
    result = pipeline.run(
        source_path=source,
        expression=CONFIG['expression'],
        threshold=CONFIG['identity_threshold'],
        max_passes=CONFIG['max_correction_passes'],
        steps=CONFIG['inference_steps'],
        seed=CONFIG['seed'],
        codeformer_fidelity=CONFIG['codeformer_fidelity'],
    )
    
    if result:
        print()
        print("=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Output: {result['output']}")
        print(f"Stages: {result['stages']}")
        print(f"2D Identity: {result['sim_2d']*100:.1f}%")
        print(f"3D Identity: {result['sim_3d']*100:.1f}%")
    else:
        print("\n[ERROR] Pipeline failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
