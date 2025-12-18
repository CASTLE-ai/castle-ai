"""
API 調用結構圖
────────────────────────────────────────────────────────────
                           create_extract_ui
                                   │
        ┌──────────────────────────┴──────────────────────────┐
        │                                                     │
  UI "Apply" 按鈕                                        UI "Extract" / "Extract Crop Video" / "Extract Rotation Latent"
        │                                                     │
        ▼                                                     ▼
 setting_preprocess                                  ┌─────────────────────────────────┐
        │                                           │ extract_roi_latent               │
        │                                           │ extract_roi_crop_video           │
        │                                           │ extract_rotation_latent          │
        ▼                                           └─────────────────────────────────┘
  Preprocess (類別)                                          │
        │                                                  ▼
        │                                      ┌─────────────────────────────────┐
        │                                      │ generate_dinov2 (observer)       │
        │                                      └─────────────────────────────────┘
        │                                                  │
        │                                                  ▼
        │                                   ┌─────────────────────────────────┐
        │                                   │  extract_roi_latent_from_video   │
        │                                   │  _extract_roi_crop_video         │
        │                                   │  extract_roi_rotation_latent_from_video  │
        │                                   └─────────────────────────────────┘
        │                                                  │
        │                                                  ▼
        │                                    Observer.extract_batch_latent /
        │                                    Observer.extract_image_latent
        │                                                  │
        └──────────────────────────────────────────────────┘

其他輔助函數：
  - init_select_video_list: 初始化影片選單
  - load_project_config: 統一讀取專案設定與目錄路徑
────────────────────────────────────────────────────────────
"""

import os
# 強制關閉 HDF5 文件鎖定 (解決 Resource temporarily unavailable 問題)
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import gradio as gr
from tqdm import tqdm
from castle import generate_dinov2, generate_dinov3
from castle.utils.video_io import ReadArray, WriteArray
from castle.utils.h5_io import H5IO
from castle.utils.video_align import (
    center_roi, rotate_based_on_roi_closest_center_point,
    crop, blank_page, rotate_based_on_deg
)
from castle.utils.plot import generate_mix_image

# ==========================================
#  核心類別：支援多核心的 Dataset (Lazy Loading)
# ==========================================
class VideoDataset(Dataset):
    def __init__(self, video_path, video_len, mask_path, preprocess, select_roi, rotate_deg=None):
        # 我們只存「路徑」，不存物件，避免多行程打架
        self.video_path = video_path
        self.video_len = video_len
        self.mask_path = mask_path
        self.preprocess = preprocess
        self.select_roi = select_roi
        self.rotate_deg = rotate_deg 
        
        # 初始化設為 None，等 Worker 自己打開
        self.reader = None 
        self.tracker = None

    def __len__(self):
        return self.video_len

    def __getitem__(self, idx):
        # Worker 第一次工作時，才打開自己的檔案
        if self.reader is None:
            self.reader = ReadArray(self.video_path)
            
        if self.tracker is None:
            # 重新開啟 H5 檔案讀取 Mask
            self.tracker = H5IO(self.mask_path) 

        frame = self.reader[idx]
        mask = self.tracker.read_mask(idx)
        
        if self.rotate_deg is not None:
             pf, pm = self.preprocess.transform(frame, mask, self.rotate_deg)
        else:
             pf, pm = self.preprocess.transform(frame, mask)
             
        return pf, pm

# ---------------------------
# 輔助函數：處理 AssertionError 並顯示 Gradio Warning
# ---------------------------
def handle_assertion_error(func):
    """裝飾器：捕獲 AssertionError 和 ValueError 並顯示 Gradio Warning"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (AssertionError, ValueError) as e:
            error_msg = str(e) if e.args else "發生了一個錯誤"
            gr.Warning(f"錯誤: {error_msg}")
            raise
    return wrapper

# ---------------------------
# 輔助函數：載入專案設定
# ---------------------------
def load_project_config(storage_path, project_name):
    project_path = os.path.join(storage_path, project_name)
    config_path = os.path.join(project_path, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    latent_dir_path = os.path.join(project_path, 'latent')
    os.makedirs(latent_dir_path, exist_ok=True)
    return project_path, config, config_path, latent_dir_path

# ---------------------------
# 初始化影片下拉選單
# ---------------------------
def init_select_video_list(storage_path, project_name):
    if not project_name:
        return gr.update(choices=[])
    project_path, config, _, _ = load_project_config(storage_path, project_name)
    if 'source' not in config:
        return gr.update(choices=[])
    choices = sorted(config['source'])
    choices.append("All")
    return gr.update(choices=choices)

# ---------------------------
# ROI Crop Video 提取
# ---------------------------
def _extract_roi_crop_video(out_path, observer, source_video, tracker, select_roi, preprocess, progress):
    fps = source_video.fps
    crf = 15
    writer = WriteArray(out_path, fps, crf)
    last_good_mask = None
    for i in tqdm(range(len(source_video)), desc="Cropping video"):
        try:
            frame = source_video[i]
            mask = tracker.read_mask(i)
            processed_frame, _ = preprocess.transform(frame, mask)
            writer.append(processed_frame)
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            h, w = frame.shape[:2]
            writer.append(blank_page(h, w))

    writer.close()
    return True

@handle_assertion_error
def extract_roi_crop_video(storage_path, project_name, select_model, select_roi, select_video, batch_size, preprocess, skip_existing, progress=gr.Progress(track_tqdm=True)):
    select_roi = int(select_roi)
    project_path, config, _, latent_dir_path = load_project_config(storage_path, project_name)
    
    # 根據選擇的模型生成對應的 observer
    if select_model == "dinov2_vitb14_reg4_pretrain":
        observer = generate_dinov2(model_type='dinov2_vitb14_reg')
    elif select_model.startswith("dinov3_"):
        # DINOv3 模型名稱直接作為 model_type
        try:
            observer = generate_dinov3(model_type=select_model, notify_func=gr.Info)
        except (ValueError, FileNotFoundError) as e:
            gr.Warning(f"無法加載 DINOv3 模型 '{select_model}': {str(e)}")
            raise
    else:
        # 默認使用 DINOv2
        observer = generate_dinov2(model_type='dinov2_vitb14_reg')
    
    config['observer_dim'] = observer.n_feature

    video_list = sorted(config['source']) if select_video == "All" else [select_video]
    out_video_paths = []
    
    # --- Pre-filter videos to get actual count for progress bar ---
    videos_to_process = []
    skipped_count = 0

    for video_name_check in video_list:
        base_name_check = os.path.splitext(video_name_check)[0]
        out_video_path_check = os.path.join(latent_dir_path, f'{base_name_check}_ROI_{select_roi}_crop.mp4')
        
        if skip_existing and os.path.exists(out_video_path_check):
            skipped_count += 1
            continue # Skip this video for processing
        
        track_dir_path_check = os.path.join(project_path, 'track', video_name_check)
        mask_list_path_check = os.path.join(track_dir_path_check, 'mask_list.h5')
        if not os.path.exists(mask_list_path_check):
            print(f"    -> WARNING: Mask file not found for {video_name_check}, skipping.")
            skipped_count += 1 # Count this as skipped for processing progress
            continue # Skip this video for processing
            
        videos_to_process.append(video_name_check) # This video will be processed

    print(f"Starting crop video extraction for {len(video_list)} video(s)...")
    
    if skipped_count > 0:
        print(f"    -> INFO: Skipped {skipped_count} existing or unmasked crop video(s) before processing.")

    for video_name in tqdm(videos_to_process, desc="Overall Progress"):
        base_name = os.path.splitext(video_name)[0]
        out_video_path = os.path.join(latent_dir_path, f'{base_name}_ROI_{select_roi}_crop.mp4')
        out_video_paths.append(out_video_path)

        source_path = os.path.join(storage_path, project_name, 'sources', video_name)
        source_video = ReadArray(source_path)
        track_dir_path = os.path.join(project_path, 'track', video_name)
        mask_list_path = os.path.join(track_dir_path, 'mask_list.h5')
        tracker = H5IO(mask_list_path)
        if not _extract_roi_crop_video(out_video_path, observer, source_video, tracker, select_roi, preprocess, progress):
            gr.Info(f"_extract_roi_crop_video fail for video: {video_name}")
            
    print("✅ Crop video extraction completed.")
    return out_video_paths

# ---------------------------
# ROI Latent 提取（已優化多核心加速）
# ---------------------------
def extract_roi_latent_from_video(observer, source_video, tracker, batch_size, select_roi, preprocess, progress):
    batch_size = int(batch_size)
    
    # --- 設定加速參數 ---
    NUM_WORKERS = os.cpu_count() // 2  # 使用一半的 CPU 核心作為 workers，以平衡 CPU 負載和系統響應
    if NUM_WORKERS == 0: # Ensure at least one worker if cpu_count is 1
        NUM_WORKERS = 1
    # ------------------

    # 1. 準備路徑 (取出路徑與長度)
    video_path = source_video.path 
    video_len = len(source_video)
    mask_path = tracker.file_path 

    # 2. 建立 Dataset (傳入路徑)
    dataset = VideoDataset(video_path, video_len, mask_path, preprocess, select_roi)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    latent_list = []
    failed_batches_count = 0

    for frames, masks in tqdm(loader, total=len(loader), desc="Extracting Latent"):
        try:
            # [優化] 直接將 DataLoader 提供的 Tensor 傳遞給新的高效方法
            # 避免在主行程中進行 Tensor -> Numpy -> List 的昂貴轉換
            latent_batch = observer.extract_tensor_batch(frames, masks, select_roi)
            latent_list.extend(latent_batch)
            
        except Exception as e:
            failed_batches_count += 1
            print(f"⚠️ WARNING: Batch failed during latent extraction. Error: {e}")
            # Continue processing other batches
            pass

    if failed_batches_count > 0:
        print(f"⚠️ WARNING: {failed_batches_count} batches failed during latent extraction for this video.")

    latent_array = np.array(latent_list)
    return latent_array

@handle_assertion_error
def extract_roi_latent(storage_path, project_name, select_model, select_roi, select_video, batch_size, preprocess, skip_existing, progress=gr.Progress(track_tqdm=True)):
    select_roi = int(select_roi)
    project_path, config, config_path, latent_dir_path = load_project_config(storage_path, project_name)
    
    # 根據選擇的模型生成對應的 observer
    if select_model == "dinov2_vitb14_reg4_pretrain":
        observer = generate_dinov2(model_type='dinov2_vitb14_reg')
    elif select_model.startswith("dinov3_"):
        # DINOv3 模型名稱直接作為 model_type
        try:
            observer = generate_dinov3(model_type=select_model, notify_func=gr.Info)
        except (ValueError, FileNotFoundError) as e:
            gr.Warning(f"無法加載 DINOv3 模型 '{select_model}': {str(e)}")
            raise
    else:
        # 默認使用 DINOv2
        observer = generate_dinov2(model_type='dinov2_vitb14_reg')
    
    config['observer_dim'] = observer.n_feature

    video_list = sorted(config['source']) if select_video == "All" else [select_video]
    latent_file_list = []
    
    # --- Pre-filter videos to get actual count for progress bar ---
    videos_to_process = []
    skipped_count = 0

    for video_name_check in video_list:
        base_name_check = os.path.splitext(video_name_check)[0]
        latent_filename_check = f'{base_name_check}_ROI_{select_roi}_latent.npz'
        latent_path_check = os.path.join(latent_dir_path, latent_filename_check)
        
        # Check for existing latent file
        if skip_existing and os.path.exists(latent_path_check):
            skipped_count += 1
            # Update config even if skipped
            if latent_filename_check not in config.get('latent', {}):
                config.setdefault('latent', {})[latent_filename_check] = video_name_check
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=4)
            continue # Skip this video for processing
        
        # Check for mask file (if missing, it's also effectively skipped for processing)
        track_dir_path_check = os.path.join(project_path, 'track', video_name_check)
        mask_list_path_check = os.path.join(track_dir_path_check, 'mask_list.h5')
        if not os.path.exists(mask_list_path_check):
            print(f"    -> WARNING: Mask file not found for {video_name_check}, skipping.")
            skipped_count += 1 # Count this as skipped for processing progress
            continue # Skip this video for processing
            
        videos_to_process.append(video_name_check) # This video will be processed

    print(f"Starting latent extraction for {len(video_list)} video(s)...")
    
    # Print summary of initially skipped files
    if skipped_count > 0:
        print(f"    -> INFO: Skipped {skipped_count} existing or unmasked file(s) before processing.")

    # Now, process only the filtered videos with an accurate overall progress bar
    for video_name in tqdm(videos_to_process, desc="Overall Progress"):
        base_name = os.path.splitext(video_name)[0]
        latent_filename = f'{base_name}_ROI_{select_roi}_latent.npz'
        latent_path = os.path.join(latent_dir_path, latent_filename)
        latent_file_list.append(latent_path) # Append only if actually processing

        source_path = os.path.join(storage_path, project_name, 'sources', video_name)
        source_video = ReadArray(source_path)
        
        track_dir_path = os.path.join(project_path, 'track', video_name)
        mask_list_path = os.path.join(track_dir_path, 'mask_list.h5')
        tracker = H5IO(mask_list_path) # This is safe to call as we pre-filtered for existence
        
        latent = extract_roi_latent_from_video(observer, source_video, tracker, batch_size, select_roi, preprocess, progress)
        
        np.savez_compressed(latent_path, latent=latent)
        config.setdefault('latent', {})[latent_filename] = video_name
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    print("✅ Latent extraction completed.")
    return latent_file_list

# ---------------------------
# ROI Rotation Latent 提取
# ---------------------------
def extract_roi_rotation_latent_from_video(observer, source_video, tracker, batch_size, select_roi, preprocess, progress):
    latent_list = []
    batch_size = int(batch_size)
    total_frames = len(source_video)
    embed_dim = observer.n_feature  # 使用 observer 的實際 embedding 維度
    num_rotations = 24  # 360 / 15 = 24 個旋轉角度
    for i in tqdm(range(0, total_frames, batch_size), desc="Extracting Rotation Latent"):
        frames, masks = [], []
        for j in range(batch_size):
            idx = i + j
            
            if idx >= total_frames:
                break
            frame = source_video[idx]
            mask = tracker.read_mask(idx)
            # 對每個 frame 依據不同旋轉角度進行處理
            for deg in range(0, 360, 15):
                pf, pm = preprocess.transform(frame, mask, deg)
                frames.append(pf)
                masks.append(pm)
        
        if not frames: # If no frames were collected in this batch (e.g., end of video)
            continue

        try:
            latent_batch = observer.extract_batch_latent(frames, masks, select_roi)
            latent_batch = np.array(latent_batch)

            if len(latent_batch) == 0:
                print(f"⚠️ WARNING: Batch latent extraction returned empty results for frames starting at {i}. Skipping this batch.")
                continue
            
            if len(latent_batch) % num_rotations != 0:
                print(f"❌ ERROR: Latent batch length ({len(latent_batch)}) is not a multiple of num_rotations ({num_rotations}) for frames starting at {i}. This batch might be malformed. Skipping this batch.")
                continue

            # 使用實際的 embedding 維度，而不是硬編碼的 768
            num_original_frames_in_batch = len(latent_batch) // num_rotations
            latent_reshaped = latent_batch.reshape(num_original_frames_in_batch, num_rotations, embed_dim)
            latent_averaged = latent_reshaped.mean(axis=1)
            latent_list.extend(latent_averaged)
        except Exception as e:
            print(f"❌ ERROR: Failed to extract latent for rotation batch starting at frame {i}. Error: {e}")
            # Skip this batch but continue with the next
            continue
            
    latent_array = np.array(latent_list)
    print('Extracted rotation latent shape:', latent_array.shape)
    return latent_array

@handle_assertion_error
def extract_rotation_latent(storage_path, project_name, select_model, select_roi, select_video, batch_size, preprocess, skip_existing, progress=gr.Progress(track_tqdm=True)):
    select_roi = int(select_roi)
    project_path, config, config_path, latent_dir_path = load_project_config(storage_path, project_name)
    
    # 根據選擇的模型生成對應的 observer
    if select_model == "dinov2_vitb14_reg4_pretrain":
        observer = generate_dinov2(model_type='dinov2_vitb14_reg')
    elif select_model.startswith("dinov3_"):
        # DINOv3 模型名稱直接作為 model_type
        try:
            observer = generate_dinov3(model_type=select_model, notify_func=gr.Info)
        except (ValueError, FileNotFoundError) as e:
            gr.Warning(f"無法加載 DINOv3 模型 '{select_model}': {str(e)}")
            raise
    else:
        # 默認使用 DINOv2
        observer = generate_dinov2(model_type='dinov2_vitb14_reg')
    
    config['observer_dim'] = observer.n_feature

    video_list = sorted(config['source']) if select_video == "All" else [select_video]
    latent_file_list = []
    
    # --- Pre-filter videos to get actual count for progress bar ---
    videos_to_process = []
    skipped_count = 0

    for video_name_check in video_list:
        base_name_check = os.path.splitext(video_name_check)[0]
        latent_filename_check = f'{base_name_check}_ROI_{select_roi}_rotation_latent.npz'
        latent_path_check = os.path.join(latent_dir_path, latent_filename_check)

        if skip_existing and os.path.exists(latent_path_check):
            skipped_count += 1
            if latent_filename_check not in config.get('latent', {}):
                config.setdefault('latent', {})[latent_filename_check] = video_name_check
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=4)
            continue # Skip this video for processing
        
        track_dir_path_check = os.path.join(project_path, 'track', video_name_check)
        mask_list_path_check = os.path.join(track_dir_path_check, 'mask_list.h5')
        if not os.path.exists(mask_list_path_check):
            print(f"    -> WARNING: Mask file not found for {video_name_check}, skipping.")
            skipped_count += 1 # Count this as skipped for processing progress
            continue # Skip this video for processing
            
        videos_to_process.append(video_name_check) # This video will be processed

    print(f"Starting rotation latent extraction for {len(video_list)} video(s)...")
    
    if skipped_count > 0:
        print(f"    -> INFO: Skipped {skipped_count} existing or unmasked rotation latent file(s) before processing.")

    for video_name in tqdm(videos_to_process, desc="Overall Progress"):
        base_name = os.path.splitext(video_name)[0]
        latent_filename = f'{base_name}_ROI_{select_roi}_rotation_latent.npz'
        latent_path = os.path.join(latent_dir_path, latent_filename)
        latent_file_list.append(latent_path)

        source_path = os.path.join(storage_path, project_name, 'sources', video_name)
        source_video = ReadArray(source_path)
        track_dir_path = os.path.join(project_path, 'track', video_name)
        mask_list_path = os.path.join(track_dir_path, 'mask_list.h5')
        tracker = H5IO(mask_list_path)

        latent = extract_roi_rotation_latent_from_video(observer, source_video, tracker, batch_size, select_roi, preprocess, progress)
        
        np.savez_compressed(latent_path, latent=latent)
        config.setdefault('latent', {})[latent_filename] = video_name
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    print("✅ Rotation latent extraction completed.")
    return latent_file_list

# ---------------------------
# 預處理類別
# ---------------------------
class Preprocess:
    def __init__(self, center_roi_switch, center_roi_id, center_roi_crop_width, center_roi_crop_height, rotate_roi_tail_switch, rotate_roi_tail_id, remove_background_switch='False'):
        self.center_roi_switch = (center_roi_switch == 'True')
        self.center_roi_id = center_roi_id
        self.center_roi_crop_width = int(center_roi_crop_width)
        self.center_roi_crop_height = int(center_roi_crop_height)
        self.rotate_roi_tail_switch = (rotate_roi_tail_switch == 'True')
        self.rotate_roi_tail_id = rotate_roi_tail_id
        self.remove_background_switch = (remove_background_switch == 'True')

    def transform(self, frame, mask, deg=0):
        try:
            if self.center_roi_switch:
                f = center_roi(frame, mask, self.center_roi_id)
                m = center_roi(mask, mask, self.center_roi_id)
                if self.rotate_roi_tail_switch:
                    f = rotate_based_on_roi_closest_center_point(f, m, self.rotate_roi_tail_id)
                    m = rotate_based_on_roi_closest_center_point(m, m, self.rotate_roi_tail_id)
                if deg > 0:
                    f = rotate_based_on_deg(f, deg)
                    m = rotate_based_on_deg(m, deg)
                f = crop(f, self.center_roi_crop_height, self.center_roi_crop_width)
                m = crop(m, self.center_roi_crop_height, self.center_roi_crop_width)
            else:
                f, m = frame, mask
            if self.remove_background_switch:
                f[m == 0] = 255
        except Exception as e:
            print(f"ERROR: Preprocessing transform failed for ROI ID {self.center_roi_id} (Center) and {self.rotate_roi_tail_id} (Tail). Error: {e}")
            f = blank_page(self.center_roi_crop_height, self.center_roi_crop_width)
            m = blank_page(self.center_roi_crop_height, self.center_roi_crop_width)
        return f, m

@handle_assertion_error
def setting_preprocess(storage_path, project_name, select_video, center_roi_switch, center_roi_id,
                       center_roi_crop_width, center_roi_crop_height, rotate_roi_tail_switch, rotate_roi_tail_id, remove_background_switch):
    preprocess = Preprocess(center_roi_switch, center_roi_id, center_roi_crop_width,
                            center_roi_crop_height, rotate_roi_tail_switch, rotate_roi_tail_id, remove_background_switch)
    project_path, config, _, _ = load_project_config(storage_path, project_name)
    video_list = sorted(config['source']) if select_video == "All" else [select_video]
    first_video = video_list[0]
    source_video = ReadArray(os.path.join(storage_path, project_name, 'sources', first_video))
    track_dir_path = os.path.join(project_path, 'track', first_video)
    mask_list_path = os.path.join(track_dir_path, 'mask_list.h5')
    tracker = H5IO(mask_list_path)
    frame, mask = source_video[0], tracker.read_mask(0)
    processed_frame, processed_mask = preprocess.transform(frame, mask)
    mixed_image = generate_mix_image(processed_frame, processed_mask)
    del tracker
    return preprocess, mixed_image

# ---------------------------
# 建立 Gradio UI
# ---------------------------
def create_extract_ui(storage_path, project_name, extract_tab):
    ui = {}
    preprocess_state = gr.State(None)
    with gr.Row(visible=True):
        with gr.Column(scale=2):
            ui['select_model'] = gr.Dropdown(
                label="Select Visual Model",
                choices=[
                    "dinov2_vitb14_reg4_pretrain",
                    "dinov3_vitb16",
                    "dinov3_vitl16",
                ],
                value="dinov2_vitb14_reg4_pretrain",
                visible=False
            )
            ui['select_roi_id'] = gr.Textbox(
                label="Enter ROI ID", value="1", info="ex: 1,2,3.", visible=False
            )
            ui['batch_size'] = gr.Textbox(
                label="Batch size", value="32", info="ex: 8, 16, 32, 64 ... Higher = faster (if GPU has enough memory)", visible=False
            )
            ui['select_video'] = gr.Dropdown(
                label="Select Target Video", value='All', visible=False
            )
            ui['skip_existing'] = gr.Checkbox(
                label="Skip existing files", value=True, interactive=True, visible=False
            )
        with gr.Column(scale=2):
            ui['center_roi_switch'] = gr.Dropdown(
                label="Center ROI", value='False', choices=['True', 'False'], visible=False
            )
            ui['center_roi_id'] = gr.Number(label="Center ROI ID", value=1, visible=False)
            ui['center_roi_crop_width'] = gr.Number(label="width", value=300, visible=False)
            ui['center_roi_crop_height'] = gr.Number(label="height", value=300, visible=False)
            ui['rotate_roi_tail_switch'] = gr.Dropdown(
                label="Rotate based on Tail", value='False', choices=['True', 'False'], visible=False
            )
            ui['rotate_roi_tail_id'] = gr.Number(label="Tail ROI ID", value=2, visible=False)
            ui['remove_background_switch'] = gr.Dropdown(
                label="Remove Background", value='False', choices=['True', 'False'], visible=False
            )
            ui['apply_preprocess'] = gr.Button("Apply", visible=False)
        with gr.Column(scale=4):
            ui['display'] = gr.Image(label='Display', interactive=False, visible=False)
            ui['extract_btn'] = gr.Button("Extract", visible=False)
            ui['extract_crop_video_btn'] = gr.Button("Extract Crop Video", visible=False)
            ui['extract_rotation_latent_btn'] = gr.Button("Extract Rotation Latent", visible=False)
            ui['latent_file_list'] = gr.File(label="ROI Visual Representation File List", visible=False)

    extract_tab.select(
        fn=init_select_video_list,
        inputs=[storage_path, project_name],
        outputs=ui['select_video']
    )
    ui['extract_crop_video_btn'].click(
        fn=extract_roi_crop_video,
        inputs=[storage_path, project_name, ui['select_model'], ui['select_roi_id'],
                ui['select_video'], ui['batch_size'], preprocess_state, ui['skip_existing']],
        outputs=ui['latent_file_list']
    )
    ui['extract_btn'].click(
        fn=extract_roi_latent,
        inputs=[storage_path, project_name, ui['select_model'], ui['select_roi_id'],
                ui['select_video'], ui['batch_size'], preprocess_state, ui['skip_existing']],
        outputs=ui['latent_file_list']
    )
    ui['extract_rotation_latent_btn'].click(
        fn=extract_rotation_latent,
        inputs=[storage_path, project_name, ui['select_model'], ui['select_roi_id'],
                ui['select_video'], ui['batch_size'], preprocess_state, ui['skip_existing']],
        outputs=ui['latent_file_list']
    )
    ui['apply_preprocess'].click(
        fn=setting_preprocess,
        inputs=[storage_path, project_name, ui['select_video'], ui['center_roi_switch'],
                ui['center_roi_id'], ui['center_roi_crop_width'], ui['center_roi_crop_height'],
                ui['rotate_roi_tail_switch'], ui['rotate_roi_tail_id'], ui['remove_background_switch']],
        outputs=[preprocess_state, ui['display']]
    )
    return ui