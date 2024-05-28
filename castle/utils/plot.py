import cv2
import numpy as np
from PIL import Image

_palette_hex = ['#7AE4F0', '#FFD0EC', '#6EE368', '#C1B5EA', '#9E83E3', '#A7CCED', '#FBC471']

_palette = [0,0,0]
for hex_code in _palette_hex:
    r, g, b = int(hex_code[1:3], 16), int(hex_code[3:5], 16), int(hex_code[5:7], 16)
    _palette.extend([r, g, b])
    

def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)

def generate_mix_image(frame, mask, alpha=0.5):
    mix = np.array(frame)
    binary_mask = (mask != 0)
    foreground = frame * (1-alpha) + colorize_mask(mask) * alpha
    mix[binary_mask] = foreground[binary_mask]
    return mix.astype(np.uint8)

def generate_mask_image(mask):
    return colorize_mask(mask).astype(np.uint8)



def generate_image_with_dots(image, dots, dots_mode):
    overlay = image.copy()
    points = np.array(dots)
    modes = np.array(dots_mode)
    neg_points = points[np.argwhere(modes == 0)[:, 0]]
    pos_points = points[np.argwhere(modes == 1)[:, 0]]

    for i in range(len(neg_points)):
        point = neg_points[i]
        cv2.circle(overlay, (point[0], point[1]), 2, (255, 80, 80), -1)

    for i in range(len(pos_points)):
        point = pos_points[i]
        cv2.circle(overlay, (point[0], point[1]), 2, (0, 153, 255), -1)

    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
    
    return image

