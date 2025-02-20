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
from castle import generate_dinov2
from castle.utils.video_io import ReadArray, WriteArray
from castle.utils.h5_io import H5IO
from castle.utils.video_align import (
    center_roi, rotate_based_on_roi_closest_center_point,
    crop, blank_page, rotate_based_on_deg
)
from castle.utils.plot import generate_mix_image

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
    for i in progress.tqdm(range(len(source_video))):
        try:
            frame = source_video[i]
            mask = tracker.read_mask(i)
            processed_frame, _ = preprocess.transform(frame, mask)
            writer.append(processed_frame)
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            writer.close()
            return False
    writer.close()
    return True

def extract_roi_crop_video(storage_path, project_name, select_model, select_roi, select_video, batch_size, preprocess, progress=gr.Progress()):
    select_roi = int(select_roi)
    project_path, config, _, latent_dir_path = load_project_config(storage_path, project_name)
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
def extract_roi_latent_from_video(observer, source_video, tracker, batch_size, select_roi, preprocess, progress):
    latent_list = []
    batch_size = int(batch_size)
    total_frames = len(source_video)
    for i in progress.tqdm(range(0, total_frames, batch_size)):
        frames, masks = [], []
        for j in range(batch_size):
            idx = i + j
            if idx >= total_frames:
                break
            frames.append(source_video[idx])
            masks.append(tracker.read_mask(idx))
        processed_frames, processed_masks = [], []
        for frame, mask in zip(frames, masks):
            pf, pm = preprocess.transform(frame, mask)
            processed_frames.append(pf)
            processed_masks.append(pm)
        try:
            latent_batch = observer.extract_batch_latent(processed_frames, processed_masks, select_roi)
            latent_list.extend(latent_batch)
        except Exception as e:
            print(f"Batch starting at frame {i} failed: {e}. Process individually.")
            for idx, (pf, pm) in enumerate(zip(processed_frames, processed_masks)):
                try:
                    latent = observer.extract_image_latent(pf, pm, select_roi)
                except Exception as ex:
                    print(f"Failed at frame {i+idx}: {ex}")
                    latent = observer.nan_latent()
                latent_list.append(latent)
    latent_array = np.array(latent_list)
    print('Extracted latent shape:', latent_array.shape)
    return latent_array

def extract_roi_latent(storage_path, project_name, select_model, select_roi, select_video, batch_size, preprocess, progress=gr.Progress()):
    select_roi = int(select_roi)
    project_path, config, config_path, latent_dir_path = load_project_config(storage_path, project_name)
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
    for i in progress.tqdm(range(0, total_frames, batch_size)):
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
        latent_batch = observer.extract_batch_latent(frames, masks, select_roi)
        latent_batch = np.array(latent_batch)
        latent_batch = latent_batch.reshape(len(latent_batch) // 24, 24, 768).mean(axis=1)
        latent_list.extend(latent_batch)
    latent_array = np.array(latent_list)
    print('Extracted rotation latent shape:', latent_array.shape)
    return latent_array

def extract_rotation_latent(storage_path, project_name, select_model, select_roi, select_video, batch_size, preprocess, progress=gr.Progress()):
    select_roi = int(select_roi)
    project_path, config, config_path, latent_dir_path = load_project_config(storage_path, project_name)
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
            print(f"Error in Preprocess.transform: {e}")
            f = blank_page(self.center_roi_crop_height, self.center_roi_crop_width)
            m = blank_page(self.center_roi_crop_height, self.center_roi_crop_width)
        return f, m

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
                choices=["dinov2_vitb14_reg4_pretrain"],
                value="dinov2_vitb14_reg4_pretrain",
                visible=False
            )
            ui['select_roi_id'] = gr.Textbox(
                label="Enter ROI ID", value="1", info="ex: 1,2,3.", visible=False
            )
            ui['batch_size'] = gr.Textbox(
                label="Batch size", value="8", info="ex: 1, 16, 64 ...", visible=False
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
