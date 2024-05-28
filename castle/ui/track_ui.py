import os

import json
import h5py
import time
import glob
import gradio as gr
import numpy as np
from natsort import natsorted
from castle import generate_aot
from castle.utils.plot import generate_mix_image
from castle.utils.h5_io import H5IO

# from api.segmentor import merge_frame_and_mask

# from api.tracker import get_aot
# from media.tracking_io import TrackingIO

# model_config = json.load(open('config/model_config.json', 'r'))


def read_label(storage_path, project_name, source_video):
    if source_video == None:
        return []
    project_path = os.path.join(storage_path, project_name)
    video_name = source_video.video_name
    label_dir_path = os.path.join(project_path, 'label', video_name)
    label_file_list = natsorted(glob.glob(f'{label_dir_path}/*.npz'))
    print('label_file_list', label_file_list)
    label_list = []
    for it in label_file_list:
        index = os.path.basename(it).split('.')[0]
        f = np.load(it)
        frame, mask = f['frame'], f['mask']
        label = dict()
        label['index'] = index
        label['frame'] = frame
        label['mask'] = mask

        label_list.append(label)

    return label_list


def read_label_to_gallery(storage_path, project_name, source_video):
    label_list = read_label(storage_path, project_name, source_video)
    gallery_list = []
    for it in label_list:
        index = it['index']
        frame, mask = it['frame'], it['mask']
        mix = generate_mix_image(frame, mask)
        gallery_list.append((mix, index))

    return label_list, gallery_list


# def setting_start_frame(label_list, evt: gr.SelectData):
#     label_index = evt.index
#     frame_index = label_list[label_index]['index']
#     return frame_index


# def collapse_accordion():
#     return gr.update(open=False)


# def open_accordion():
#     return gr.update(open=True)

# def init_display(label_list, evt: gr.SelectData):
#     label_index = evt.index
#     frame, mask = label_list[label_index]['frame'], label_list[label_index]['mask']
#     mix = generate_mix_image(frame, mask)
#     return mix



class Interfence:
    def __init__(self, storage_path, project_name, source_video, start, stop, max_len, model_aot):
        self.cancel = False
        self.show_middle_result = False
        self.model_aot = model_aot

        project_path = os.path.join(storage_path, project_name)
        video_name = source_video.video_name
        track_dir_path = os.path.join(project_path, 'track', video_name)
        os.makedirs(track_dir_path, exist_ok=True)
        self.track_dir_path = track_dir_path

        label_path = os.path.join(project_path, 'label', video_name, f'{start}.npz')
        # self.first_label = np.load(label_path)
        self.source_video = source_video
        self.start = int(start)
        self.stop = int(stop)
        self.max_len = int(max_len)

        self.knowledges = []
        label_list = read_label(storage_path, project_name, source_video)
        self.roi_count = 0
        for it in label_list:
            index = it['index']
            frame, mask = it['frame'], it['mask']
            self.knowledges.append((frame, mask))
            self.roi_count = max(self.roi_count, np.max(mask))

        
        pass



    def tracking(self, progress):
        
        time.sleep(1)
        start, stop, max_len = self.start, self.stop, self.max_len,
        tracker = generate_aot(model_type=self.model_aot)
        # self.roi_count = np.max(self.first_label['mask'])
        self.write_mask_config()
        for f, m in self.knowledges:
            tracker.add_reference_frame(f, m, self.roi_count, -1)


        # tracker.add_reference_frame(self.first_label['frame'], self.first_label['mask'], self.roi_count, -1)
        delta = 1 if start < stop else -1
  

        for i in progress.tqdm(range(start, stop + delta, delta)):
            if self.cancel:
                self.show_middle_result = False
                self.cancel = False
                return "Cancel"
            
            frame = self.source_video[i]
            mask = tracker.track(frame)
            tracker.update_memory(mask)
            self.frame, self.mask = frame, mask.squeeze().detach().cpu().numpy().astype(np.uint8)
            self.write_mask(i, self.mask)

        self.show_middle_result = False
        return "Done"
    
    def write_mask_config(self):
        mask_list_path = os.path.join(self.track_dir_path, f'mask_list.h5')       


        tracker = H5IO(mask_list_path)
        tracker.write_config('roi_count', self.roi_count)
        tracker.write_config('total_frames', len(self.source_video))
        tracker.write_config('height', self.source_video.video_stream.height)
        tracker.write_config('width', self.source_video.video_stream.width)
        del tracker


    def write_mask(self, index, mask):
        mask_list_path = os.path.join(self.track_dir_path, f'mask_list.h5')            
        mode = 'a' if os.path.isfile(mask_list_path) else 'w'
        with h5py.File(mask_list_path, mode) as f:
            if f"{index}" in f:
                dset = f[f"{index}"][:]
                dset[:] = mask
            else:
                dset = f.create_dataset(f"{index}", mask.shape, dtype='uint8', compression="gzip", compression_opts=9)
                dset[:] = mask


    def set_cancel(self):
        self.cancel = True
        pass

    def flip_show_middle_result(self):
        if self.show_middle_result:
            self.show_middle_result = False
        else:
            self.show_middle_result = True


def init_Interfence(storage_path, project_name, source_video, start, stop, max_len, model_aot):
    print('init_Interfence', start, stop)
    return Interfence(storage_path, project_name, source_video, start, stop, max_len, model_aot)

def run_interfence(interfence, progress=gr.Progress()):
    status = interfence.tracking(progress)
    return f"{status}. From {interfence.start} to {interfence.stop}"

def click_middle_result(interfence):
    print('enter click_middle_result', interfence.show_middle_result)
    interfence.flip_show_middle_result()
    while interfence.show_middle_result:
        time.sleep(1)
        yield generate_mix_image(interfence.frame, interfence.mask), display_middle_result_mode(interfence.show_middle_result)

def display_middle_result_mode(res):
    if res:
       return "Show"
    else:
       return "Close"

def set_cancel(interfence):
    interfence.set_cancel()



def create_track_ui(storage_path, project_name, source_video, track_tab):
    ui = dict()

    label_list = gr.State(None)
    interfence = gr.State(None)
    with gr.Accordion('ROIs Knowledge', visible=False) as ui['gallery_accordion']:
        ui['gallery'] = gr.Gallery(
            label="Label Frame", show_label=True, allow_preview=False, object_fit="contain", columns=3)

    with gr.Accordion('Inference', open=True, visible=False) as ui['inference_accordion']:
        with gr.Row(visible=True):
            with gr.Column(scale=2):
                # ui['start_frame'] = gr.Textbox(
                #     label="Start Frame", interactive=False, visible=False)
                ui['start_frame'] = gr.Slider(
                    label="Start Frame (include)", minimum=0, step=1, maximum=1, value=0, interactive=True, visible=False)
                ui['stop_frame'] = gr.Slider(
                    label="Stop Frame (include)", minimum=0, step=1, maximum=1, value=1, interactive=True, visible=False)
                ui['deaot_model'] = gr.Dropdown(['r50_deaotl', 'swinb_deaotl'], label='tracking_model',
                              info='swim_transformer or res50',
                              value='r50_deaotl',
                              interactive=True)
                ui['long_term_max_len'] = gr.Number(
                    label="Long term menary length",
                    info="Bigger is better, but this depends on the GPU's RAM capacity",
                    value=30, interactive=True, visible=False
                )
                ui['init_tracker'] = gr.Button(
                    "init Tracker", interactive=True, visible=False)
                ui['tracking_btm'] = gr.Button(
                    "Tracking ROIs", interactive=True, visible=False)
                ui['progress_edit'] = gr.Textbox(
                    label="Progress", visible=False)
                ui['display_middle_result_mode'] = gr.Textbox(value="Close",
                    label="Display Mode", interactive=False, visible=False)
                ui['display_middle_result'] = gr.Button(
                    "Display middle result", interactive=True, visible=False)
                ui['cancel_btn'] = gr.Button(
                    "Cancel", interactive=True, visible=False)

            with gr.Column(scale=8):
                ui['display'] = gr.Image(
                    label='Display', interactive=False, visible=False)


    tracking_config = [ui['start_frame'], ui['stop_frame'], ui['long_term_max_len'], ui['deaot_model']]
    track_tab.select(
        fn=read_label_to_gallery,
        inputs=[storage_path, project_name, source_video],
        outputs=[label_list, ui['gallery']]
    )

    # ui['gallery'].select(
    #     fn=setting_start_frame,
    #     inputs=[label_list],
    #     outputs=ui['start_frame']
    # )
    # ui['gallery'].select(
    #     fn=collapse_accordion,
    #     outputs=ui['gallery_accordion']
    # )
    # ui['gallery'].select(
    #     fn=open_accordion,
    #     outputs=ui['inference_accordion']
    # )

    # ui['gallery'].select(
    #     fn=init_display,
    #     inputs=label_list,
    #     outputs=ui['display']
    # )

    init_Interfence_input = [storage_path, project_name, source_video]
    init_Interfence_input.extend(tracking_config)
    ui['init_tracker'].click(
        fn=init_Interfence,
        inputs=init_Interfence_input,
        outputs= interfence
    )

    ui['tracking_btm'].click(
        fn=run_interfence,
        inputs = interfence,
        outputs=ui['progress_edit']
    )

    ui['display_middle_result'].click(
        fn=click_middle_result,
        inputs=interfence,
        outputs=[ui['display'], ui['display_middle_result_mode']]
    )

    ui['cancel_btn'].click(
        fn=set_cancel,
        inputs=interfence,
    )

    return ui
