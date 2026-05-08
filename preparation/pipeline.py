import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

@dataclass
class MaskConfig:
    blur_kernel: tuple = (11, 11)
    canny_low: int = 5
    canny_high: int = 60
    
    close_kernel: tuple = (11, 11)
    close_iterations: int = 8
    dilate_iterations: int = 1
    
    border_thickness: int = 10
    smooth_blur: tuple = (21, 21)
    open_kernel: tuple = (5, 5)
    open_iterations: int = 1
    
    grabcut_iterations: int = 3

CONFIG = MaskConfig()

def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = np.zeros_like(mask)
    if num_labels <= 1:
        return mask.copy(), 0
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_area = int(stats[largest_label, cv2.CC_STAT_AREA])
    result[labels == largest_label] = 255
    return result, largest_area

def fill_holes(mask):
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    ffmask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 255)
    inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, inv)
    return filled[1:-1, 1:-1]

def build_rope_mask(img_bgr, keep_main_component=True):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, CONFIG.blur_kernel, 0)
    
    edges = cv2.Canny(blurred, CONFIG.canny_low, CONFIG.canny_high, apertureSize=3, L2gradient=True)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CONFIG.close_kernel)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=CONFIG.close_iterations)
    edges_dilated = cv2.dilate(edges_closed, kernel_close, iterations=CONFIG.dilate_iterations)
    
    mask_filled = fill_holes(edges_dilated)
    
    h, w = mask_filled.shape
    cv2.rectangle(mask_filled, (0, 0), (w-1, h-1), 0, thickness=CONFIG.border_thickness)
    
    mask_blurred = cv2.GaussianBlur(mask_filled, CONFIG.smooth_blur, 0)
    _, mask_smooth = cv2.threshold(mask_blurred, 127, 255, cv2.THRESH_BINARY)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CONFIG.open_kernel)
    mask_clean = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel_open, iterations=CONFIG.open_iterations)
    
    if keep_main_component:
        mask_main, _ = keep_largest_component(mask_clean)
    else:
        mask_main = mask_clean.copy()

    return mask_main

def refine_mask_with_grabcut(img_crop, mask_crop):
    gc_mask = np.where(mask_crop > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype('uint8')
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_crop, gc_mask, None, bgdModel, fgdModel, CONFIG.grabcut_iterations, cv2.GC_INIT_WITH_MASK)
    except Exception:
        pass 

    refined_mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255, 0
    ).astype('uint8')

    refined_mask, _ = keep_largest_component(refined_mask)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_closed = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel_close, None, None, 1)

    return mask_closed

def find_knot_roi(mask, min_side=220, density_thresh_factor=0.8):
    mask_main, _ = keep_largest_component(mask)
    temp = fill_holes(mask_main)
    binary = (temp > 0).astype('uint8')

    h_img, w_img = binary.shape[:2]
    k = int(w_img * 0.2)
    if k % 2 == 0: k += 1
        
    density = cv2.blur(binary.astype('float32'), (k, k))
    max_den = float(density.max())
    if max_den == 0.0:
        raise RuntimeError("Leere Maske: Keine weißen Pixel gefunden.")

    thr = max_den * density_thresh_factor
    hot = np.zeros_like(binary, dtype='uint8')
    hot[density >= thr] = 255

    ys, xs = np.where(hot > 0)
    if len(ys) == 0:
        ys, xs = np.where(mask_main > 0)

    x0, y0 = xs.min(), ys.min()
    x1, y1 = xs.max(), ys.max()

    cx = int((x0 + x1) / 2)
    cy = int((y0 + y1) / 2)

    ww = x1 - x0
    hh = y1 - y0
    side = int(max(ww, hh) * 1.2)
    side = max(side, min_side)

    h_img, w_img = mask.shape[:2]
    x1_box = max(0, cx - side // 2)
    y1_box = max(0, cy - side // 2)
    x2_box = min(w_img, cx + side // 2)
    y2_box = min(h_img, cy + side // 2)

    roi = (x1_box, y1_box, x2_box - x1_box, y2_box - y1_box, cx, cy)
    return roi

def prepare_for_cnn(knot_crop, knot_mask, target_size=224):
    gray = cv2.cvtColor(knot_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    knot_no_bg = cv2.bitwise_and(enhanced, enhanced, mask=knot_mask)

    h, w = knot_no_bg.shape[:2]
    max_side = max(h, w)
    top = (max_side - h) // 2
    bottom = max_side - h - top
    left = (max_side - w) // 2
    right = max_side - w - left

    padded = cv2.copyMakeBorder(knot_no_bg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    cnn_ready = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return cnn_ready

def process_image(img_path_or_array):
    if isinstance(img_path_or_array, (str, Path)):
        img = cv2.imread(str(img_path_or_array))
        if img is None:
            raise ValueError(f"Bild konnte nicht geladen werden: {img_path_or_array}")
    else:
        img = img_path_or_array

    try:
        h, w = img.shape[:2]
        max_width = 1200 
        
        if w > max_width:
            scale = max_width / float(w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        mask = build_rope_mask(img)
        roi = find_knot_roi(mask, min_side=224, density_thresh_factor=0.6)
        
        x, y, w_roi, h_roi, cx, cy = roi
        knot_crop = img[y:y + h_roi, x:x + w_roi]
        knot_mask_rough = mask[y:y + h_roi, x:x + w_roi]
        
        if knot_crop.size == 0 or knot_mask_rough.size == 0:
            raise ValueError("Ungültige ROI (leeres Crop). Knoten nicht gefunden.")

        knot_mask_refined = refine_mask_with_grabcut(knot_crop, knot_mask_rough)
        cnn_image = prepare_for_cnn(knot_crop, knot_mask_refined, target_size=224)

        return cnn_image
        
    except Exception as e:
        print(f"Fehler in der Pipeline: {e}")
        return None