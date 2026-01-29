class UltraRealisticFaceSwap:
    def __init__(self, models_root=None, buffalo_root=None, inswapper_path=None):
        """Initialize face analyzer and swapper models. Auto-detects platform."""
        import platform
        is_linux = platform.system() == 'Linux'
        
        # Set default paths based on platform
        if models_root is None:
            models_root = '/teamspace/studios/this_studio/Magicface' if is_linux else r'd:\Magicface_gravity'
        self.models_root = models_root
        
        # Buffalo_l root  
        if buffalo_root is None:
            buffalo_root = '/teamspace/studios/this_studio/Magicface/Model/models' if is_linux else os.path.join(models_root, 'models')
        
        # Inswapper path
        if inswapper_path is None:
            inswapper_path = '/teamspace/studios/this_studio/models/inswapper_128.onnx' if is_linux else os.path.join(models_root, 'models', 'inswapper_128.onnx')
        
        print(f"Platform: {'Linux/Cloud' if is_linux else 'Windows'}")
        print(f"Buffalo_l root: {buffalo_root}")
        print(f"Inswapper path: {inswapper_path}")
        
        print("Loading face analyzer (buffalo_l)...")
        self.app = FaceAnalysis(
            name='buffalo_l',
            root=buffalo_root,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.3)
        
        if not os.path.exists(inswapper_path):
            raise FileNotFoundError(f"Inswapper model not found: {inswapper_path}")
        
        print(f"Loading inswapper from: {inswapper_path}")
        self.swapper = insightface.model_zoo.get_model(inswapper_path)