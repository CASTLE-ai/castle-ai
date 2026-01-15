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
import json
import numpy as np
import gradio as gr
from castle import generate_dinov2, generate_dinov3
from castle.utils.video_io import ReadArray, WriteArray
from castle.utils.h5_io import H5IO
from castle.utils.video_align import (
    center_roi, rotate_based_on_roi_closest_center_point,
    crop, blank_page, rotate_based_on_deg,
    get_roi_closest_point_safe, rotate_based_on_point
)
from castle.utils.plot import generate_mix_image

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
    for i in progress.tqdm(range(len(source_video))):
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
def extract_roi_crop_video(storage_path, project_name, select_model, select_roi, select_video, batch_size, preprocess, progress=gr.Progress()):
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

    # 決定要處理的影片
    video_list = sorted(config['source']) if select_video == "All" else [select_video]

    out_video_path = ""
    for video_name in video_list:
        source_path = os.path.join(storage_path, project_name, 'sources', video_name)
        source_video = ReadArray(source_path)
        track_dir_path = os.path.join(project_path, 'track', video_name)
        mask_list_path = os.path.join(track_dir_path, 'mask_list.h5')
        tracker = H5IO(mask_list_path)
        base_name = os.path.splitext(source_video.video_name)[0]
        out_video_path = os.path.join(latent_dir_path, f'{base_name}_ROI_{select_roi}_crop.mp4')
        if not _extract_roi_crop_video(out_video_path, observer, source_video, tracker, select_roi, preprocess, progress):
            gr.Info(f"_extract_roi_crop_video fail for video: {video_name}")
    return out_video_path

# ---------------------------
# ROI Latent 提取（一般模式）
# ---------------------------
def interpolate_missing_points(valid_points, total_frames):
    """
    對缺失的幀執行線性內插或外插
    
    Args:
        valid_points: dict {frame_idx: (x, y)} 有效的追蹤點
        total_frames: 總幀數
    
    Returns:
        dict {frame_idx: (x, y)} 包含所有幀的點（內插/外插後）
    """
    if not valid_points:
        raise ValueError("No valid tracking points found for rotate_roi_tail_id")
    
    result = {}
    
    for idx in range(total_frames):
        if idx in valid_points:
            result[idx] = valid_points[idx]
        else:
            # 找前後最近的有效點
            prev_indices = [i for i in valid_points.keys() if i < idx]
            next_indices = [i for i in valid_points.keys() if i > idx]
            prev_idx = max(prev_indices) if prev_indices else None
            next_idx = min(next_indices) if next_indices else None
            
            if prev_idx is not None and next_idx is not None:
                # 線性內插
                t = (idx - prev_idx) / (next_idx - prev_idx)
                prev_point = valid_points[prev_idx]
                next_point = valid_points[next_idx]
                result[idx] = (
                    prev_point[0] + t * (next_point[0] - prev_point[0]),
                    prev_point[1] + t * (next_point[1] - prev_point[1])
                )
            elif prev_idx is not None:
                # 外插（使用前一個有效點）
                print(f"Warning: Extrapolating at end of video for frame {idx} using frame {prev_idx}")
                result[idx] = valid_points[prev_idx]
            elif next_idx is not None:
                # 外插（使用後一個有效點）
                print(f"Warning: Extrapolating at beginning of video for frame {idx} using frame {next_idx}")
                result[idx] = valid_points[next_idx]
            else:
                # 不應該發生（因為 valid_points 不為空）
                raise ValueError(f"Cannot interpolate/extrapolate for frame {idx}")
    
    return result


def extract_roi_latent_from_video(observer, source_video, tracker, batch_size, select_roi, preprocess, progress):
    latent_list = []
    batch_size = int(batch_size)
    total_frames = len(source_video)
    
    # 預先處理：如果需要旋轉，先掃描所有幀收集有效的 closest points  
    interpolated_points = None
    if preprocess.rotate_roi_tail_switch:
        print(f"Scanning all frames for rotate_roi_tail_id={preprocess.rotate_roi_tail_id}...")
        valid_points = {}
        failed_frames = []
        for idx in progress.tqdm(range(total_frames), desc="Scanning ROI"):
            try:
                mask = tracker.read_mask(idx)
                # 需要先將 mask center 到 center_roi_id，因為 transform 會這樣做
                if preprocess.center_roi_switch:
                    mask = center_roi(mask, mask, preprocess.center_roi_id)
                point = get_roi_closest_point_safe(mask, preprocess.rotate_roi_tail_id)
                if point is not None:
                    valid_points[idx] = point
                else:
                    failed_frames.append(idx)
            except Exception as e:
                # 如果 center_roi 失敗（例如 center_roi_id 不存在），跳過這一幀
                # 這些幀會在後續通過內插來處理
                failed_frames.append(idx)
        
        print(f"Found {len(valid_points)}/{total_frames} frames with valid tracking")
        if failed_frames:
            # 找出連續的失敗區間
            failed_regions = []
            if failed_frames:
                start = failed_frames[0]
                end = failed_frames[0]
                for i in range(1, len(failed_frames)):
                    if failed_frames[i] == end + 1:
                        end = failed_frames[i]
                    else:
                        failed_regions.append((start, end))
                        start = failed_frames[i]
                        end = failed_frames[i]
                failed_regions.append((start, end))
            
            print(f"Failed frames in {len(failed_regions)} regions:")
            for start, end in failed_regions[:5]:  # 只顯示前5個區間
                print(f"  Frames {start}-{end} ({end-start+1} frames)")
            if len(failed_regions) > 5:
                print(f"  ... and {len(failed_regions)-5} more regions")
        
        # 執行內插/外插
        interpolated_points = interpolate_missing_points(valid_points, total_frames)
    
    # 主處理迴圈
    for i in progress.tqdm(range(0, total_frames, batch_size), desc="Extracting latent"):
        frames, masks = [], []
        for j in range(batch_size):
            idx = i + j
            if idx >= total_frames:
                break
            frames.append(source_video[idx])
            masks.append(tracker.read_mask(idx))
        
        processed_frames, processed_masks = [], []
        for j, (frame, mask) in enumerate(zip(frames, masks)):
            idx = i + j
            # 如果有內插的點，傳入 transform
            closest_point = interpolated_points[idx] if interpolated_points else None
            pf, pm = preprocess.transform(frame, mask, precomputed_closest_point=closest_point)
            processed_frames.append(pf)
            processed_masks.append(pm)
        
        try:
            latent_batch = observer.extract_batch_latent(processed_frames, processed_masks, select_roi)
            latent_list.extend(latent_batch)
        except Exception as e:
            print(f"Batch starting at frame {i} failed: {e}. Process individually.")
            for j, (pf, pm) in enumerate(zip(processed_frames, processed_masks)):
                idx = i + j
                try:
                    latent = observer.extract_image_latent(pf, pm, select_roi)
                except Exception as ex:
                    print(f"Failed at frame {idx}: {ex}")
                    latent = observer.nan_latent()
                latent_list.append(latent)
    
    latent_array = np.array(latent_list)
    print('Extracted latent shape:', latent_array.shape)
    return latent_array

@handle_assertion_error
def extract_roi_latent(storage_path, project_name, select_model, select_roi, select_video, batch_size, preprocess, progress=gr.Progress()):
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
    for video_name in video_list:
        source_path = os.path.join(storage_path, project_name, 'sources', video_name)
        source_video = ReadArray(source_path)
        track_dir_path = os.path.join(project_path, 'track', video_name)
        mask_list_path = os.path.join(track_dir_path, 'mask_list.h5')
        tracker = H5IO(mask_list_path)
        latent = extract_roi_latent_from_video(observer, source_video, tracker, batch_size, select_roi, preprocess, progress)
        base_name = os.path.splitext(source_video.video_name)[0]
        latent_path = os.path.join(latent_dir_path, f'{base_name}_ROI_{select_roi}_latent.npz')
        np.savez_compressed(latent_path, latent=latent)
        config.setdefault('latent', {})[f'{base_name}_ROI_{select_roi}_latent.npz'] = source_video.video_name
        with open(config_path, 'w') as f:
            json.dump(config, f)
        latent_file_list.append(latent_path)
    return latent_file_list

# ---------------------------
# ROI Rotation Latent 提取
# ---------------------------
def extract_roi_rotation_latent_from_video(observer, source_video, tracker, batch_size, select_roi, preprocess, progress):
    latent_list = []
    batch_size = int(batch_size)
    total_frames = len(source_video)
    embed_dim = observer.n_feature  # 使用 observer 的實際 embedding 維度
    num_rotations = 7  # 360 / 15 = 24 個旋轉角度
    for i in progress.tqdm(range(0, total_frames, batch_size)):
        frames, masks = [], []
        for j in range(batch_size):
            idx = i + j
            
            if idx >= total_frames:
                break
            frame = source_video[idx]
            mask = tracker.read_mask(idx)
            # 對每個 frame 依據不同旋轉角度進行處理
            for deg in np.arange(0, 360, (360/num_rotations)):
                pf, pm = preprocess.transform(frame, mask, deg)
                frames.append(pf)
                masks.append(pm)
        latent_batch = observer.extract_batch_latent(frames, masks, select_roi)
        latent_batch = np.array(latent_batch)
        # 使用實際的 embedding 維度，而不是硬編碼的 768
        latent_batch = latent_batch.reshape(len(latent_batch) // num_rotations, num_rotations, embed_dim).mean(axis=1)
        latent_list.extend(latent_batch)
    latent_array = np.array(latent_list)
    print('Extracted rotation latent shape:', latent_array.shape)
    return latent_array

@handle_assertion_error
def extract_rotation_latent(storage_path, project_name, select_model, select_roi, select_video, batch_size, preprocess, progress=gr.Progress()):
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
    for video_name in video_list:
        source_path = os.path.join(storage_path, project_name, 'sources', video_name)
        source_video = ReadArray(source_path)
        track_dir_path = os.path.join(project_path, 'track', video_name)
        mask_list_path = os.path.join(track_dir_path, 'mask_list.h5')
        tracker = H5IO(mask_list_path)
        latent = extract_roi_rotation_latent_from_video(observer, source_video, tracker, batch_size, select_roi, preprocess, progress)
        base_name = os.path.splitext(source_video.video_name)[0]
        latent_path = os.path.join(latent_dir_path, f'{base_name}_ROI_{select_roi}_rotation_latent.npz')
        np.savez_compressed(latent_path, latent=latent)
        config.setdefault('latent', {})[f'{base_name}_ROI_{select_roi}_rotation_latent.npz'] = source_video.video_name
        with open(config_path, 'w') as f:
            json.dump(config, f)
        latent_file_list.append(latent_path)
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

    def transform(self, frame, mask, deg=0, precomputed_closest_point=None):
        try:
            if self.center_roi_switch:
                f = center_roi(frame, mask, self.center_roi_id)
                m = center_roi(mask, mask, self.center_roi_id)
                if self.rotate_roi_tail_switch:
                    if precomputed_closest_point is not None:
                        # 使用預先計算（可能是內插）的點
                        try:
                            f = rotate_based_on_point(f, precomputed_closest_point)
                            m = rotate_based_on_point(m, precomputed_closest_point)
                        except Exception as rot_e:
                            print(f"Error in rotate_based_on_point with precomputed point {precomputed_closest_point}: {rot_e}")
                            raise
                    else:
                        # 原有邏輯：直接從 mask 計算
                        f = rotate_based_on_roi_closest_center_point(f, m, self.rotate_roi_tail_id)
                        m = rotate_based_on_roi_closest_center_point(m, m, self.rotate_roi_tail_id)
            else:
                f, m = frame, mask

            if not deg == 0:
                f = rotate_based_on_deg(f, deg)
                m = rotate_based_on_deg(m, deg)

            if self.center_roi_switch:
                f = crop(f, self.center_roi_crop_height, self.center_roi_crop_width)
                m = crop(m, self.center_roi_crop_height, self.center_roi_crop_width)
            if self.remove_background_switch:
                f[m == 0] = 255
        except Exception as e:
            print(f"Error in Preprocess.transform: {e}")
            print(f"  center_roi_switch={self.center_roi_switch}, rotate_roi_tail_switch={self.rotate_roi_tail_switch}")
            print(f"  precomputed_closest_point={precomputed_closest_point}")
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
                ui['select_video'], ui['batch_size'], preprocess_state],
        outputs=ui['latent_file_list']
    )
    ui['extract_btn'].click(
        fn=extract_roi_latent,
        inputs=[storage_path, project_name, ui['select_model'], ui['select_roi_id'],
                ui['select_video'], ui['batch_size'], preprocess_state],
        outputs=ui['latent_file_list']
    )
    ui['extract_rotation_latent_btn'].click(
        fn=extract_rotation_latent,
        inputs=[storage_path, project_name, ui['select_model'], ui['select_roi_id'],
                ui['select_video'], ui['batch_size'], preprocess_state],
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
