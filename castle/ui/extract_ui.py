import gradio as gr
import os
import json
import numpy as np
# from api.roi_observer import DinoV2latentGen
from castle import generate_dinov2
# from media.source_video import SourceVideo
from castle.utils.video_io import ReadArray
import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as tt

from castle.utils.h5_io import H5IO
resolution = 518
patch_len = resolution // 14
img2tensor = tt.Compose([
    tt.ToTensor(), # range [0, 255] -> [0.0,1.0]
    tt.Resize((resolution, resolution), antialias=True),
    tt.Normalize(mean=0.5, std=0.2), # range [0.0,1.0] -> [-2.5, 2.5]

])

img2mask = tt.Compose([
    tt.ToTensor(), # range [0, 255] -> [0.0,1.0]
    tt.Resize((resolution, resolution), antialias=True),
])


model_config = json.load(open('config/model_config.json', 'r'))

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


def extract_roi_latent(storage_path, project_name, select_model, select_roi, select_video, batch_size, progress=gr.Progress()):
    batch_size = int(batch_size)
    project_path = os.path.join(storage_path, project_name)
    project_config_path = os.path.join(project_path, 'config.json')
    project_config = json.load(open(project_config_path, 'r'))
    latent_dir_path = os.path.join(storage_path, project_name, 'latent')
    os.makedirs(latent_dir_path, exist_ok=True)
    observer = generate_dinov2(model_config['dinov2_args'])
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
        # f = np.load(mask_list_path)
        # mask_list = f['mask_list']
        tracker = H5IO(mask_list_path)
        try:
            latent = extract_roi_latent_from_video(observer, source_video, tracker, batch_size, select_roi, progress)
        except:
            gr.Info("latent extract fail")
            del observer
            return []
        base_name = source_video.video_name.split('.')[0]
        latent_path = os.path.join(latent_dir_path, f'{base_name}_ROI_{select_roi}_latent.npz')
        print('latent', type(latent))
        np.savez_compressed(latent_path, latent=latent)

        if not 'latent' in project_config:
            project_config['latent'] = dict()
        
        project_config['latent'][f'{base_name}_ROI_{select_roi}_latent.npz'] = source_video.video_name
        json.dump(project_config, open(project_config_path,'w'))
        latent_file_list.append(latent_path)
    del observer
    del tracker
    return latent_file_list


class CustomImageDataset(Dataset):
    def __init__(self, source_video, tracker, select_roi):
        self.source_video = source_video
        self.tracker = tracker
        self.select_roi = select_roi

    def __len__(self):
        return self.tracker.read_config('total_frames')

    def __getitem__(self, index):
        return img2tensor(self.source_video.read_by_index(index).to_rgb().to_ndarray()), img2mask(self.tracker.read_mask(index) == self.select_roi)
    


def extract_roi_latent_from_video(observer, source_video, tracker, batch_size, select_roi, progress):
    # select_roi = torch.tensor(select_roi)
    latent_list = []
    batch_size = int(batch_size)
    print('batch_size', (batch_size))
    dataset = CustomImageDataset(source_video, tracker, select_roi)
    print('dataset', len(dataset))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    # print('dataloader', len(dataloader))
    # print('observer.n_feature', observer.n_feature)
    print('select_roi', select_roi, type(select_roi))

    for i, (frames, masks) in enumerate(progress.tqdm(dataloader)):
        patch_feature = observer.batch_run(frames)

        for img, mask in zip(patch_feature, masks):
            latent = img.reshape((patch_len, patch_len, observer.n_feature))
            small_mask = mask.reshape(resolution//14, 14, resolution//14, 14).sum(axis=(1, 3))
            sum_mask = small_mask.sum()

            result = small_mask[:, :, np.newaxis] * latent
            latent_mask_ave = result.sum(axis=0).sum(axis=0) / sum_mask
            latent_list.append(latent_mask_ave)


    latent_list = np.array(latent_list)
    print('latent_list', latent_list.shape)
    return latent_list


def create_extract_ui(storage_path, project_name, extract_tab):
    ui = dict()
    with gr.Row(visible=True):
        with gr.Column(scale=2):
            ui['select_model'] = gr.Dropdown(
                label="Select Visual Model", 
                choices=["dinov2_vitb14_reg4_pretrain"], value="dinov2_vitb14_reg4_pretrain", visible=False)
            ui['select_roi_id'] = gr.Textbox(label="Enter ROI ID", value="1", info="ex: 1,2,3.", visible=False)
            ui['batch_size'] = gr.Textbox(label="Batch size", value="16", info="ex: 1, 16, 64 ...", visible=False)
            ui['select_video'] = gr.Dropdown(label="Select Target Video", visible=False)
            ui['extract_btn'] = gr.Button("Extract", visible=False)
            # ui['progress_bar'] = gr.Textbox(label="Progress", visible=False)
        with gr.Column(scale=8):
            ui['latent_file_list'] = gr.File(label="ROI Visual Representation File List", visible=False)

    extract_tab.select(
        fn=init_select_video_list,
        inputs=[storage_path, project_name],
        outputs=ui['select_video']
    )

    ui['extract_btn'].click(
        fn=extract_roi_latent,
        inputs=[storage_path, project_name, ui['select_model'], ui['select_roi_id'], ui['select_video'], ui['batch_size']],
        outputs=ui['latent_file_list']
    )
    return ui