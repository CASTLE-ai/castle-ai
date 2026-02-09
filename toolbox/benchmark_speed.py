#!/home/isonaei/ABVFM_benchmark/venvs/castle/bin/python
import sys
import os
import time
import argparse
import json
import torch
import cv2
import numpy as np

# Ensure 'castle' package can be found if running from toolbox or root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    from castle.utils.video_object_segment import generate_aot
    from castle.core.models import get_visual_encoder
except ImportError:
    # Fallback if running from root without package installed in editable mode
    sys.path.append(os.getcwd())
    from castle.utils.video_object_segment import generate_aot
    from castle.core.models import get_visual_encoder


def load_frames(video_path, max_frames=None):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at {video_path}")

    # print(f"Loading frames from {video_path}...", file=sys.stderr)
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV uses BGR, models usually expect RGB or handle it internally.
        # Consistency: Convert to RGB for general purpose usage in encoders
        # DeAOT might handle BGR, but let's standardize on RGB for loaded frames if possible,
        # OR keep as BGR and convert only when needed. 
        # Original scripts: 
        # benchmark_deaot kept BGR (DeAOT usually expects BGR/RGB via transforms)
        # benchmark_all converted to RGB.
        # We will return list of numpy arrays (BGR for consistency with cv2, convert inside specific funcs if needed)
        # Actually benchmark_all converted to RGB. Let's convert to RGB here to be safe for DINO.
        # DeAOT implementation in castle usually handles transforms.
        
        # Let's check original benchmark_deaot: it just appended 'frame' (BGR).
        # benchmark_all: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).
        
        # To support both without double conversion, let's keep as BGR here and documented.
        # WAIT: DINOv2 encoder in castle likely expects RGB. 
        # Let's convert to RGB to be safe for modern models, DeAOT wrapper should handle it.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    # print(f"Loaded {len(frames)} frames.", file=sys.stderr)
    return frames, fps

def calculate_time_ratio(processing_time, num_frames, video_fps=30.0):
    if processing_time <= 0:
        return 0.0
    video_duration = num_frames / video_fps
    return processing_time / video_duration

def benchmark_deaot(model_type, frames, batch_size=4):
    result = {
        "model": model_type,
        "frames": len(frames),
        "batch_size": batch_size,
        "total_time": 0.0,
        "fps": 0.0,
        "time_ratio": 0.0,
        "error": None
    }
    
    try:
        # Load Model
        tracker = generate_aot(model_type=model_type)
        
        # Setup Reference Frame
        h, w = frames[0].shape[:2]
        dummy_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(dummy_mask, (w//2, h//2), 30, 1, -1)
        
        # DeAOT expects numpy arrays. Our frames are RGB. 
        # If the underlying model expects BGR, this might be a slight color shift but for speed benchmark it doesn't matter.
        
        tracker.add_reference_frame(frames[0], dummy_mask, obj_nums=1, frame_step=-1)
        
        start_run = time.time()
        
        # Inference
        for i in range(1, len(frames), batch_size):
            batch_frames = frames[i : i+batch_size]
            tensors = []
            orig_sizes = []
            for f in batch_frames:
                sample = {'current_img': f}
                sample = tracker.transform(sample)
                tensors.append(sample[0]['current_img'])
                orig_sizes.append(f.shape[:2])
            
            if not tensors: break
            
            batch_tensor = torch.stack(tensors)
            tracker.track_batch(batch_tensor, original_sizes=orig_sizes)
            
        end_run = time.time()
        result["total_time"] = end_run - start_run
        
        processed_count = len(frames) - 1 # Exclude Ref
        if result["total_time"] > 0:
            result["fps"] = processed_count / result["total_time"]
            result["time_ratio"] = result["total_time"] / (processed_count / 30.0) # Assuming 30fps for ratio consistency
            
    except Exception as e:
        result["error"] = str(e)
        
    return result

def benchmark_encoder(model_name, frames, batch_size=16):
    result = {
        "model": model_name,
        "frames": len(frames),
        "batch_size": batch_size,
        "total_time": 0.0,
        "fps": 0.0,
        "time_ratio": 0.0,
        "error": None
    }
    
    try:
        encoder = get_visual_encoder(model_name)
        encoder.load_model()
        
        h, w = frames[0].shape[:2]
        dummy_mask = np.ones((h, w), dtype=np.uint8)
        
        # Warmup
        encoder.extract_tensor_batch([frames[0]], [dummy_mask], roi_id=1)
        
        start_run = time.time()
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i+batch_size]
            batch_masks = [dummy_mask] * len(batch_frames)
            encoder.extract_tensor_batch(batch_frames, batch_masks, roi_id=1)
            
        end_run = time.time()
        result["total_time"] = end_run - start_run
        
        if result["total_time"] > 0:
            result["fps"] = len(frames) / result["total_time"]
            result["time_ratio"] = result["total_time"] / (len(frames) / 30.0)

    except Exception as e:
        result["error"] = str(e)
        
    return result

def main():
    parser = argparse.ArgumentParser(description="Benchmark CASTLE models speed.")
    parser.add_argument("--video_path", type=str, default="projects/2026-01-05-17-00-40-Project_ctrl_30fps.mp4/sources/ctrl_30fps.mp4", help="Path to video file")
    parser.add_argument("--model", type=str, choices=['all', 'deaot', 'dinov2', 'dinov3'], default='all', help="Model to benchmark")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames to load")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (for encoders, DeAOT defaults to 4 internally usually but can be overridden if logic allows)")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    
    args = parser.parse_args()
    
    results = {}
    
    try:
        # Load frames once
        frames, vid_fps = load_frames(args.video_path, args.frames)
        results["video_info"] = {
            "path": args.video_path,
            "total_frames_loaded": len(frames),
            "video_fps": vid_fps
        }
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return

    # Benchmark DeAOT
    if args.model in ['all', 'deaot']:
        # DeAOT usually uses smaller batch size due to memory
        deaot_bs = 4 if args.batch_size > 4 else args.batch_size
        res = benchmark_deaot('r50_deaotl', frames, batch_size=deaot_bs)
        results['deaot'] = res

    # Benchmark DINOv2
    if args.model in ['all', 'dinov2']:
        res = benchmark_encoder('dinov2_vitb14', frames, batch_size=args.batch_size)
        results['dinov2'] = res
        
    # Benchmark DINOv3
    if args.model in ['all', 'dinov3']:
        torch.cuda.empty_cache()
        res = benchmark_encoder('dinov3_vitb16', frames, batch_size=args.batch_size)
        results['dinov3'] = res

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\n=== Benchmark Results ({args.video_path}) ===")
        print(f"Frames Loaded: {len(frames)}")
        
        for key, res in results.items():
            if key == "video_info": continue
            print(f"\n--- {key.upper()} ---")
            if res.get("error"):
                print(f"Error: {res['error']}")
                continue
                
            print(f"Total Time: {res['total_time']:.2f}s")
            print(f"FPS: {res['fps']:.2f}")
            print(f"Time Ratio (Proc/Real): {res['time_ratio']:.2f}x")
            print(f"  (>1.0 means slower than realtime, <1.0 means faster)")


if __name__ == "__main__":
    main()
