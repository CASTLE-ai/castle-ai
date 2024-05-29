import gradio as gr
import os
import json
import numpy as np

from castle import generate_dinov2
from castle.utils.video_io import ReadArray
import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as tt

from castle.utils.h5_io import H5IO
from castle.utils.video_align import center_roi, rotate_based_on_roi_closest_center_point, crop, blank_page
from castle.utils.plot import generate_mix_image

def init_select_video_list(storage_path, project_name):
    if project_name == None:
        return gr.update(choices=[])
    project_path = os.path.join(storage_path, project_name)
    project_config_path = os.path.join(project_path, 'config.json')
    project_config = json.load(open(project_config_path, 'r'))
    if not 'source' in project_config:
         return gr.update(choices=[])
    
    subdirectories = sorted(project_config['source'])
    subdirectories.append("All")
    return gr.update(choices=subdirectories)


def extract_roi_latent(storage_path, project_name, select_model, select_roi, select_video, batch_size, preprocess, progress=gr.Progress()):
    batch_size = int(batch_size)
    project_path = os.path.join(storage_path, project_name)
    project_config_path = os.path.join(project_path, 'config.json')
    project_config = json.load(open(project_config_path, 'r'))
    latent_dir_path = os.path.join(storage_path, project_name, 'latent')
    os.makedirs(latent_dir_path, exist_ok=True)
    observer = generate_dinov2(model_type='dinov2_vitb14_reg')
    project_config['observer_dim'] = observer.n_feature
    latent_file_list = []
    select_roi = int(select_roi)
    
    if select_video == "All":
        subdirectories = sorted(project_config['source'])

    else:
        subdirectories = [select_video]

    for it in subdirectories:
        source_video = ReadArray(os.path.join(storage_path, project_name, 'sources', it))
        track_dir_path = os.path.join(project_path, 'track', it)
        mask_list_path = os.path.join(track_dir_path, f'mask_list.h5')
        tracker = H5IO(mask_list_path)
        latent = extract_roi_latent_from_video(observer, source_video, tracker, batch_size, select_roi, preprocess, progress)
        # try:
        #     latent = extract_roi_latent_from_video(observer, source_video, tracker, batch_size, select_roi, progress)
        # except:
        #     gr.Info("latent extract fail")
        #     del observer
        #     return []
        base_name = source_video.video_name.split('.')[0]
        latent_path = os.path.join(latent_dir_path, f'{base_name}_ROI_{select_roi}_latent.npz')
        # print('latent', type(latent))
        np.savez_compressed(latent_path, latent=latent)

        if not 'latent' in project_config:
            project_config['latent'] = dict()
        
        project_config['latent'][f'{base_name}_ROI_{select_roi}_latent.npz'] = source_video.video_name
        json.dump(project_config, open(project_config_path,'w'))
        latent_file_list.append(latent_path)
    del observer
    del tracker
    return latent_file_list


# class CustomImageDataset(Dataset):
#     def __init__(self, source_video, tracker):
#         self.source_video = source_video
#         self.tracker = tracker

#     def __len__(self):
#         return min(self.tracker.read_config('total_frames'), len(self.source_video))

#     def __getitem__(self, index):
#         return self.source_video[index], self.tracker.read_mask(index)
    


def extract_roi_latent_from_video(observer, source_video, tracker, batch_size, select_roi, preprocess, progress):

    latent_list = []
    batch_size = int(batch_size)
    print('batch_size', (batch_size))


    for i in progress.tqdm(range(0, len(source_video), batch_size)):
        frames = [source_video[i+j] for j in range(batch_size) if j+i < len(source_video)]
        masks = [tracker.read_mask(i+j) for j in range(batch_size) if j+i < len(source_video)]
        
        frames_process, masks_process = [], []
        for f, m in zip(frames, masks):
            ff, mm = preprocess.transform(f, m)
            frames_process.append(ff)
            masks_process.append(mm)

        frames = frames_process
        masks = masks_process

        try:
            latent = observer.extract_batch_latent(frames, masks, select_roi)
            latent_list.extend(latent)
            continue
        except:
            print(f"batch {i} error, try to run frame one by one")

        for j in range(batch_size):
            try:
                latent = observer.extract_image_latent(frames[j], masks[j], select_roi)
                latent_list.append(latent)
            except:
                latent_list.append(observer.nan_latent())
                print(f'fail at frame {i*batch_size+j}')

    latent_list = np.array(latent_list)
    print('latent_list', latent_list.shape)
    return latent_list


class Preprocess:
    def __init__(self, center_roi_switch, center_roi_id, center_roi_crop_width, center_roi_crop_height,  rotate_roi_tail_switch, rotate_roi_tail_id):
        self.center_roi_switch = center_roi_switch
        self.center_roi_id=center_roi_id
        self.center_roi_crop_width=center_roi_crop_width
        self.center_roi_crop_height=center_roi_crop_height
        self.rotate_roi_tail_switch=rotate_roi_tail_switch
        self.rotate_roi_tail_id=rotate_roi_tail_id


    def transform(self, frame, mask):

        try:
            if self.center_roi_switch == 'True':
                f = center_roi(frame, mask, self.center_roi_id)
                m = center_roi(mask, mask, self.center_roi_id)
                if self.rotate_roi_tail_switch == 'True':
                    f = rotate_based_on_roi_closest_center_point(f, m, self.rotate_roi_tail_id)
                    m = rotate_based_on_roi_closest_center_point(m, m, self.rotate_roi_tail_id)
                f = crop(f, self.center_roi_crop_height, self.center_roi_crop_width)
                m = crop(m, self.center_roi_crop_height, self.center_roi_crop_width)
            else:
                f, m = frame, mask
        except:
            f = blank_page(self.center_roi_crop_height, self.center_roi_crop_width)
            m = blank_page(self.center_roi_crop_height, self.center_roi_crop_width)

        return f, m

def setting_preprocess(storage_path, project_name, select_video, center_roi_switch, center_roi_id, center_roi_crop_width, center_roi_crop_height,  rotate_roi_tail_switch, rotate_roi_tail_id):
    preprocess = Preprocess(center_roi_switch, center_roi_id, center_roi_crop_width, center_roi_crop_height,  rotate_roi_tail_switch, rotate_roi_tail_id)

    project_path = os.path.join(storage_path, project_name)
    project_config_path = os.path.join(project_path, 'config.json')
    project_config = json.load(open(project_config_path, 'r'))
    latent_dir_path = os.path.join(storage_path, project_name, 'latent')
    os.makedirs(latent_dir_path, exist_ok=True)
    observer = generate_dinov2(model_type='dinov2_vitb14_reg')
    project_config['observer_dim'] = observer.n_feature

    if select_video == "All":
        subdirectories = sorted(project_config['source'])
    else:
        subdirectories = [select_video]
    print('subdirectories', subdirectories)
    first_frame = ReadArray(os.path.join(storage_path, project_name, 'sources', subdirectories[0]))[0]
    track_dir_path = os.path.join(project_path, 'track', subdirectories[0])
    mask_list_path = os.path.join(track_dir_path, f'mask_list.h5')
    tracker = H5IO(mask_list_path)
    f, m = preprocess.transform(first_frame, tracker.read_mask(0))

    return preprocess, generate_mix_image(f, m)


def create_extract_ui(storage_path, project_name, extract_tab):
    ui = dict()
    preprocess = gr.State(None)
    with gr.Row(visible=True):
        with gr.Column(scale=2):
            ui['select_model'] = gr.Dropdown(
                label="Select Visual Model", 
                choices=["dinov2_vitb14_reg4_pretrain"], value="dinov2_vitb14_reg4_pretrain", visible=False)
            ui['select_roi_id'] = gr.Textbox(label="Enter ROI ID", value="1", info="ex: 1,2,3.", visible=False)
            ui['batch_size'] = gr.Textbox(label="Batch size", value="16", info="ex: 1, 16, 64 ...", visible=False)
            ui['select_video'] = gr.Dropdown(label="Select Target Video", value='All', visible=False)
            
            # ui['progress_bar'] = gr.Textbox(label="Progress", visible=False)
        with gr.Column(scale=2):
            ui['center_roi_switch'] =gr.Dropdown(label="Center ROI", value='False', choices=['True', 'False'], visible=False)
            ui['center_roi_id'] =gr.Number(label="Center ROI ID", value=1, visible=False)
            ui['center_roi_crop_width'] =gr.Number(label="width", value=300, visible=False)
            ui['center_roi_crop_height'] =gr.Number(label="height", value=300, visible=False)
            ui['rotate_roi_tail_switch'] =gr.Dropdown(label="Rotate based on Tail", value='False', choices=['True', 'False'], visible=False)
            ui['rotate_roi_tail_id'] =gr.Number(label="Tail ROI ID", value=2, visible=False)
            ui['apply_preprocess'] = gr.Button("Apply", visible=False)
            
        with gr.Column(scale=4):
            ui['display'] = gr.Image(
                    label='Display', interactive=False, visible=False)
            ui['extract_btn'] = gr.Button("Extract", visible=False)
            ui['latent_file_list'] = gr.File(label="ROI Visual Representation File List", visible=False)
            

    extract_tab.select(
        fn=init_select_video_list,
        inputs=[storage_path, project_name],
        outputs=ui['select_video']
    )

    ui['extract_btn'].click(
        fn=extract_roi_latent,
        inputs=[storage_path, project_name, ui['select_model'], ui['select_roi_id'], ui['select_video'], ui['batch_size'], preprocess],
        outputs=ui['latent_file_list']
    )
    ui['apply_preprocess'].click(
        fn=setting_preprocess,
        inputs=[storage_path, project_name, ui['select_video'], ui['center_roi_switch'], ui['center_roi_id'],ui['center_roi_crop_width'],ui['center_roi_crop_height'],  ui['rotate_roi_tail_switch'],ui['rotate_roi_tail_id'] ],
        outputs=[preprocess, ui['display']]
    )
    return ui
