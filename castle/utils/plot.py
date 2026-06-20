"""Low-level plotting utilities for mask visualization."""

import cv2
import numpy as np

from castle.core.config import PALETTE_HEX as _palette_hex

_palette = [0,0,0]
for hex_code in _palette_hex:
    r, g, b = int(hex_code[1:3], 16), int(hex_code[3:5], 16), int(hex_code[5:7], 16)
    _palette.extend([r, g, b])

# (256, 3) uint8 lookup table from the palette — lets colorize_mask be a single
# vectorised numpy index instead of a PIL palette round-trip (faster, releases the
# GIL, so it scales inside the mix-video overlay thread pool). Palette has 256
# entries (256*3 ints); pad/truncate to exactly 256 rows.
_lut = np.zeros((256, 3), dtype=np.uint8)
_flat = np.array(_palette[:256 * 3], dtype=np.uint8)
_lut[:len(_flat) // 3] = _flat[:(len(_flat) // 3) * 3].reshape(-1, 3)


def colorize_mask(pred_mask):
    return _lut[pred_mask.astype(np.uint8)]

def generate_mix_image(frame, mask, alpha=0.5):
    # Alpha-blend the colorised mask over the frame, but only where mask != 0.
    # uint8 cv2.addWeighted (GIL-releasing) replaces the old float64 + PIL path.
    colorized = _lut[mask.astype(np.uint8)]
    blended = cv2.addWeighted(frame, 1.0 - alpha, colorized, alpha, 0.0)
    out = frame.copy()
    m = mask != 0
    out[m] = blended[m]
    return out

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

