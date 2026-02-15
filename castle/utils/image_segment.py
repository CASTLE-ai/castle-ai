"""SAM-based image segmentation utilities."""

import torch
import numpy as np
from .download import download_file
from castle.sam.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from castle.core.environment import get_device
DEFAULT_DEVICE = get_device()


class Segmentor:
    """SAM-based interactive image segmentor.

    Wraps the Segment Anything Model for point-and-click ROI segmentation.
    Supports both automatic mask generation and interactive point/mask prompts.

    Args:
        sam_args: Dict with 'sam_checkpoint', 'model_type', 'generator_args', 'device'.
        sam_model: Optional pre-loaded SAM model to reuse (avoids reloading weights).
    """

    def __init__(self, sam_args, sam_model=None):
        """
        sam_args:
            sam_checkpoint: path of SAM checkpoint
            generator_args: args for everything_generator
            device: device
        sam_model: optional pre-loaded SAM model to reuse (avoids reloading weights)
        """
        self.device = sam_args["device"]
        if sam_model is not None:
            self.sam = sam_model
        else:
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
    """Multi-ROI segmentor that accumulates clicks across multiple objects.

    Manages sequential point-and-click segmentation for multiple ROIs,
    assigning incrementing ROI IDs and compositing masks. Reuses a single
    SAM model instance across all clicks.
    """

    def __init__(self, sam_args, sam_model=None) -> None:
        self.sam_args = sam_args
        self.click_points = []
        self.click_modes = []
        self.n_rois = 0
        self.next = True
        # Load SAM model once and reuse across clicks
        if sam_model is not None:
            self._sam_model = sam_model
        else:
            self._sam_model = sam_model_registry[sam_args["model_type"]](
                checkpoint=sam_args["sam_checkpoint"]
            )
            self._sam_model.to(device=sam_args["device"])


    def set_frame(self, frame):
        self.frame = frame
        self.pre_mask = np.zeros(frame.shape[:2]).astype(np.uint8)

    def segment_with_click(self, point, mode):
        if self.next:
            self.n_rois += 1
            self.next = False

        self.click_points.append(point)
        self.click_modes.append(mode)

        # Reuse the cached SAM model instead of reloading every click
        sam = Segmentor(self.sam_args, sam_model=self._sam_model)
        mask = sam.segment_with_click(
            self.frame, 
            np.array(self.click_points), 
            np.array(self.click_modes)
        )
        # Don't delete sam — the underlying model is shared via self._sam_model
        self.temp_mask = np.array(self.pre_mask)
        self.temp_mask[mask > 0] = self.n_rois
        return self.temp_mask

    def next_roi(self):
        self.next = True
        self.pre_mask = self.temp_mask
        self.click_points = []
        self.click_modes = []

        

    def __del__(self):
        if hasattr(self, '_sam_model'):
            del self._sam_model
        torch.cuda.empty_cache()
   


def download_sa_ckpt(model_type):
    from castle.core.config import DEFAULT_CKPT_DIR
    
    if model_type == 'vit_b':
        ckpt_path = DEFAULT_CKPT_DIR / 'sam_vit_b_01ec64.pth'
        download_file('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', str(ckpt_path))
        return str(ckpt_path)
    else:
        assert False, f"model_type mismatch {model_type}, expect vit_b."


def load_sam_model(ckpt_path='', model_type='vit_b', device=''):
    """Load and return the raw SAM model (heavy weights).
    
    Call once and keep the result in a gr.State to avoid reloading
    the model on every frame switch.
    """
    if not ckpt_path:
        ckpt_path = download_sa_ckpt(model_type)
    if not device:
        device = DEFAULT_DEVICE
    model = sam_model_registry[model_type](checkpoint=ckpt_path)
    model.to(device=device)
    return model


def generate_sa(ckpt_path='', model_type='vit_b', device='', sam_model=None):
    """Create a MultiObjectSegmentor, optionally reusing a pre-loaded SAM model.
    
    Args:
        sam_model: Pre-loaded SAM model from load_sam_model(). If None,
                   the model is loaded from scratch.
    """
    if not ckpt_path:
        ckpt_path = download_sa_ckpt(model_type)
    if not device:
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
    return MultiObjectSegmentor(sam_args, sam_model=sam_model)