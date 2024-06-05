
import cv2
import torch
import numpy as np
from PIL import Image
from .download import download_file
from castle.sam.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import platform
OS_SYS = platform.uname().system
if OS_SYS == 'Darwin':
    DEFAULT_DEVICE = 'mps'
elif torch.cuda.is_available():
    DEFAULT_DEVICE = 'cuda'
else:
    DEFAULT_DEVICE = 'cpu'


# np.random.seed(200)
# _palette = ((np.random.random((3*255))*0.7+0.3)*255).astype(np.uint8).tolist()
# _palette = [0,0,0]+_palette

# def colorize_mask(pred_mask):
#     save_mask = Image.fromarray(pred_mask.astype(np.uint8))
#     save_mask = save_mask.convert(mode='P')
#     save_mask.putpalette(_palette)
#     save_mask = save_mask.convert(mode='RGB')
#     return np.array(save_mask)

# def draw_points(points, modes, frame):

#     overlay = frame.copy()
#     points = np.array(points)
#     modes = np.array(modes)
#     neg_points = points[np.argwhere(modes == 0)[:, 0]]
#     pos_points = points[np.argwhere(modes == 1)[:, 0]]

#     for i in range(len(neg_points)):
#         point = neg_points[i]
#         cv2.circle(overlay, (point[0], point[1]), 2, (255, 80, 80), -1)

#     for i in range(len(pos_points)):
#         point = pos_points[i]
#         cv2.circle(overlay, (point[0], point[1]), 2, (0, 153, 255), -1)

#     cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
#     return frame

# def merge_frame_and_mask(frame, mask, points = [], modes = [], alpha=0.5):
#     mix = np.array(frame)
#     binary_mask = (mask != 0)
#     foreground = frame * (1-alpha) + colorize_mask(mask) * alpha
#     mix[binary_mask] = foreground[binary_mask]
#     mix = draw_points(points, modes, mix)
#     return mix.astype(np.uint8)


# def mask2img(mask):
#     img = colorize_mask(mask)
#     return img.astype(np.uint8)

class Segmentor:
    def __init__(self, sam_args):
        """
        sam_args:
            sam_checkpoint: path of SAM checkpoint
            generator_args: args for everything_generator
            device: device
        """
        self.device = sam_args["device"]
        self.sam = sam_model_registry[sam_args["model_type"]](checkpoint=sam_args["sam_checkpoint"])
        self.sam.to(device=self.device)
        self.everything_generator = SamAutomaticMaskGenerator(model=self.sam, **sam_args['generator_args'])
        self.interactive_predictor = self.everything_generator.predictor
        self.have_embedded = False
        
    @torch.no_grad()
    def set_image(self, image):
        if not self.have_embedded:
            self.interactive_predictor.set_image(image)
            self.have_embedded = True

    @torch.no_grad()
    def interactive_predict(self, prompts, mode, multimask=True):
        assert self.have_embedded, 'image embedding for sam need be set before predict.'        
        
        if mode == 'point':
            masks, scores, logits = self.interactive_predictor.predict(point_coords=prompts['point_coords'], 
                                point_labels=prompts['point_modes'], 
                                multimask_output=multimask)
        elif mode == 'mask':
            masks, scores, logits = self.interactive_predictor.predict(mask_input=prompts['mask_prompt'], 
                                multimask_output=multimask)
        elif mode == 'point_mask':
            masks, scores, logits = self.interactive_predictor.predict(point_coords=prompts['point_coords'], 
                                point_labels=prompts['point_modes'], 
                                mask_input=prompts['mask_prompt'], 
                                multimask_output=multimask)
                                
        return masks, scores, logits
        
    @torch.no_grad()
    def segment_with_click(self, origin_frame, coords, modes, multimask=True):
        '''
            
            return: 
                mask: one-hot 
        '''
        self.set_image(origin_frame)

        prompts = {
            'point_coords': coords,
            'point_modes': modes,
        }
        masks, scores, logits = self.interactive_predict(prompts, 'point', multimask)
        mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
        prompts = {
            'point_coords': coords,
            'point_modes': modes,
            'mask_prompt': logit[None, :, :]
        }
        masks, scores, logits = self.interactive_predict(prompts, 'point_mask', multimask)
        mask = masks[np.argmax(scores)]

        return mask.astype(np.uint8)
    


class MultiObjectSegmentor():
    def __init__(self, sam_args) -> None:
        self.sam_args = sam_args
        self.click_points = []
        self.click_modes = []
        self.roi_count = 0
        self.next = True
        pass


    def set_frame(self, frame):
        self.frame = frame
        self.pre_mask = np.zeros(frame.shape[:2]).astype(np.uint8)

    def segment_with_click(self, point, mode):
        if self.next:
            self.roi_count += 1
            self.next = False

        self.click_points.append(point)
        self.click_modes.append(mode)

        sam = Segmentor(self.sam_args)
        mask = sam.segment_with_click(
            self.frame, 
            np.array(self.click_points), 
            np.array(self.click_modes)
        )
        del sam
        self.temp_mask = np.array(self.pre_mask)
        self.temp_mask[mask > 0] = self.roi_count
        return self.temp_mask

    def next_roi(self):
        self.next = True
        self.pre_mask = self.temp_mask
        self.click_points = []
        self.click_modes = []

        

    def __del__(self):
        torch.cuda.empty_cache()
   


def download_sa_ckpt(model_type):
    if model_type == 'vit_b':
        ckpt_path = 'ckpt/sam_vit_b_01ec64.pth'
        download_file('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', ckpt_path)

        return ckpt_path
    else:
        assert False, f"model_type mismatch {model_type}, expect vit_b."


def generate_sa(ckpt_path='', model_type='vit_b', device=''):
    if len(ckpt_path) == 0:
        ckpt_path = download_sa_ckpt(model_type)
    if len(device) == 0:
        device = DEFAULT_DEVICE

    sam_args = {
        'sam_checkpoint': ckpt_path,
        'model_type': model_type,
        'generator_args':{
            'points_per_side': 16,
            'pred_iou_thresh': 0.8,
            'stability_score_thresh': 0.9,
            'crop_n_layers': 1,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 200,
        },
        'device': device,
    }
    return MultiObjectSegmentor(sam_args)