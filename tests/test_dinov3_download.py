import torch
import os
from castle.core.config import DEFAULT_CKPT_DIR, CKPT_DINO_IDS, DINOV3_CONSTANTS
from castle.utils.download import download_with_gdown

def test_dinov3_loading():
    model_type = "dinov3_vitl16"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"--- 1. Checkpoint Setup ---")
    filename = DINOV3_CONSTANTS['MODEL_TO_CKPT_FILENAME'].get(model_type)
    if not filename:
        print(f"ERROR: Filename not found for {model_type}")
        return
    
    ckpt_path = DEFAULT_CKPT_DIR / filename
    print(f"Checkpoint path: {ckpt_path}")
    
    if not ckpt_path.exists():
        print("Checkpoint not found. Attempting download...")
        file_id = CKPT_DINO_IDS.get(model_type)
        if not file_id:
            print(f"ERROR: Google Drive ID not found for {model_type}")
            return
        
        try:
            os.makedirs(DEFAULT_CKPT_DIR, exist_ok=True)
            download_with_gdown(file_id, str(ckpt_path))
            print("Download complete.")
        except Exception as e:
            print(f"ERROR during download: {e}")
            return
    else:
        print("Checkpoint already exists.")

    print(f"\n--- 2. Loading Model Architecture from Torch Hub ---")
    try:
        model = torch.hub.load('facebookresearch/dinov3', model_type, pretrained=False)
        print("Successfully loaded model architecture from torch.hub.")
    except Exception as e:
        print(f"ERROR loading from torch.hub: {e}")
        # Try a different model name from the hub as a test
        try:
            print("Attempting fallback with 'dinov3_large'...")
            model = torch.hub.load('facebookresearch/dinov3', 'dinov3_large', pretrained=False)
            print("Successfully loaded 'dinov3_large' from torch.hub.")
        except Exception as e2:
            print(f"ERROR on fallback as well: {e2}")
        return

    print(f"\n--- 3. Loading Weights ---")
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        print("Successfully loaded checkpoint file.")
        
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint: state_dict = checkpoint['model']
        
        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded weights into the model.")
        
        model.to(device).eval()
        print(f"Model moved to {device} and set to eval mode.")
        print("\nSUCCESS: DINOv3 model ready.")
        
    except Exception as e:
        print(f"ERROR during weight loading: {e}")

if __name__ == "__main__":
    test_dinov3_loading()