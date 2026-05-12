import cv2
import math
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from cv2 import ximgproc


RING_OFFSETS = [
    (-1, -1), (0, -1), (1, -1),
    (1, 0), (1, 1), (0, 1),
    (-1, 1), (-1, 0)
]

def choose_best_channel(img_bgr):
    #gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    channels = {
        #'Gray': gray, 
        'H': hsv[:, :, 0], 'S': hsv[:, :, 1], 
        #'V': hsv[:, :, 2],
        #'L': lab[:, :, 0], 
        'A': lab[:, :, 1], 'B': lab[:, :, 2],
    }
    best_name, best_ch, best_score = None, None, -1.0
    for name, ch in channels.items():
        hist = cv2.calcHist([ch], [0], None, [256], [0, 256]).ravel().astype(np.float64)
        total = hist.sum()
        if total == 0: continue
        hist /= total
        thr, _ = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = int(thr)
        w0, w1 = hist[:k].sum(), hist[k:].sum()
        if w0 == 0 or w1 == 0:
            score = 0.0
        else:
            idx = np.arange(256, dtype=np.float64)
            mu0 = (idx[:k] * hist[:k]).sum() / w0
            mu1 = (idx[k:] * hist[k:]).sum() / w1
            mu_t = (idx * hist).sum()
            sigma_b = w0 * w1 * (mu0 - mu1) ** 2
            sigma_t = ((idx - mu_t) ** 2 * hist).sum()
            score = 0.0 if sigma_t == 0 else sigma_b / sigma_t
        if score > best_score:
            best_score, best_name, best_ch = score, name, ch
    return best_name, best_ch, float(best_score)

def smart_invert(mask):
    corners = [mask[0, 0], mask[0, -1], mask[-1, 0], mask[-1, -1]]
    white_corners = sum(int(v == 255) for v in corners)
    if white_corners >= 2: return cv2.bitwise_not(mask)
    return mask

def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = np.zeros_like(mask)
    if num_labels <= 1: return mask.copy(), 0
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_area = int(stats[largest_label, cv2.CC_STAT_AREA])
    result[labels == largest_label] = 255
    return result, largest_area

def build_rope_mask(img_bgr, close_k=(11,11), open_k=(6,6), close_iter=9, flat_k=151, flat_add=150):
    best_name, best_channel, best_score = choose_best_channel(img_bgr)
    _, mask_raw = cv2.threshold(best_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_raw = smart_invert(mask_raw)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_k)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_k)
    mask_closed = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel_close, None, None, close_iter)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)
    mask_main, _ = keep_largest_component(mask_clean)
    return mask_main

def thin_binary(mask):
    mask = (mask > 0).astype(np.uint8) * 255
    try:
        return cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except AttributeError:
        raise RuntimeError("cv2.ximgproc error!")

def count_neighbor_groups(sk01, x, y):
    vals = []
    h, w = sk01.shape
    for dx, dy in RING_OFFSETS:
        xx, yy = x + dx, y + dy
        if 0 <= xx < w and 0 <= yy < h: vals.append(1 if sk01[yy, xx] > 0 else 0)
        else: vals.append(0)
    groups = sum(1 for i in range(len(vals)) if vals[i] == 1 and vals[i - 1] == 0)
    return groups, int(sum(vals))


def prune_skeleton(skel, prune_iter=40):
    skel_clean = skel.copy()
                       
    for _ in range(prune_iter):
        skel_bin = (skel_clean > 0).astype(np.uint8)
        
        pad = np.pad(skel_bin, 1, mode='constant', constant_values=0)
        
        p2 = pad[0:-2, 1:-1] # Oben
        p3 = pad[0:-2, 2:]   # Oben-Rechts
        p4 = pad[1:-1, 2:]   # Rechts
        p5 = pad[2:,   2:]   # Unten-Rechts
        p6 = pad[2:,   1:-1] # Unten
        p7 = pad[2:,   0:-2] # Unten-Links
        p8 = pad[1:-1, 0:-2] # Links
        p9 = pad[0:-2, 0:-2] # Oben-Links
        
        neighbors = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
        
        crossing_number = np.zeros_like(skel_bin, dtype=np.uint8)
        for i in range(8):
            transition = (neighbors[i] == 0) & (neighbors[i+1] == 1)
            crossing_number += transition.astype(np.uint8)

        endpoints = (skel_bin == 1) & (crossing_number <= 1)
        
        if not endpoints.any():
            break
            
        skel_clean[endpoints] = 0
        
    return skel_clean

def fill_holes(mask):
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    ffmask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    
    cv2.floodFill(flood, ffmask, (0, 0), 255)
    
    inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, inv)
    return filled[1:-1, 1:-1]

def analyze_skeleton_crossings(mask, border_margin=30, region_merge_size=5, min_merge_px=20, rel_diag_factor=0.015, prune_length=40):
    mask_bin = mask.copy()

    h, w = mask_bin.shape
    
    cv2.rectangle(mask_bin, (0, 0), (w-1, h-1), 0, thickness=20)
    
    blurred = cv2.GaussianBlur(mask_bin, (21, 21), 0)
    _, mask_bin = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    k_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, k_smooth)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, k_smooth)
    
    mask_bin, _ = keep_largest_component(mask_bin)

    skel = thin_binary(mask_bin)
    
    skel = prune_skeleton(skel, prune_iter=prune_length)
    
    sk01 = (skel > 0).astype(np.uint8)
    candidate_img = np.zeros_like(skel)
    group_map = np.zeros_like(skel, dtype=np.uint8)
    neighbor_map = np.zeros_like(skel, dtype=np.uint8)

    for y in range(border_margin, h - border_margin):
        for x in range(border_margin, w - border_margin):
            if sk01[y, x] == 0: continue
            groups, n_count = count_neighbor_groups(sk01, x, y)
            group_map[y, x] = groups
            neighbor_map[y, x] = n_count
            if groups >= 3 and n_count >= 3:
                candidate_img[y, x] = 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (region_merge_size, region_merge_size))
    merged = cv2.dilate(candidate_img, k, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

    raw_crossings = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ys, xs = np.where(labels == i)
        region_cands = []
        for x, y in zip(xs, ys):
            if candidate_img[y, x] == 0: continue
            region_cands.append((x, y, int(group_map[y, x]), int(neighbor_map[y, x])))
        if not region_cands: continue
        region_cands.sort(key=lambda t: (t[2], t[3]), reverse=True)
        bx, by, bg, bn = region_cands[0]
        raw_crossings.append({'x': bx, 'y': by, 'groups': bg, 'neighbors': bn, 'region_area': area})

    diag = math.hypot(w, h)
    merge_dist = max(float(min_merge_px), float(rel_diag_factor) * diag)
    merge_dist_sq = merge_dist * merge_dist

    remaining, merged_crossings = list(raw_crossings), []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        changed = True
        while changed:
            changed, keep = False, []
            for item in remaining:
                if any((item['x']-c['x'])**2 + (item['y']-c['y'])**2 <= merge_dist_sq for c in cluster):
                    cluster.append(item)
                    changed = True
                else: keep.append(item)
            remaining = keep
        cluster.sort(key=lambda t: (t['groups'], t['neighbors'], t['region_area']), reverse=True)
        merged_crossings.append(cluster[0])

    return {'seed_skeleton': skel, 'crossing_centers_merged': [(c['x'], c['y']) for c in merged_crossings]}

def make_overlay(image_bgr, mask, skeleton, crossings):
    overlay = image_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    ys, xs = np.where(skeleton > 0)
    overlay[ys, xs] = (0, 0, 255)

    for i, (cx, cy) in enumerate(crossings):
        cv2.circle(overlay, (cx, cy), 25, (0, 255, 255), 4)
        cv2.circle(overlay, (cx, cy), 6, (255, 0, 0), -1)
        cv2.putText(overlay, str(i + 1), (cx + 20, cy - 20), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 0, 255), 3)

    return overlay


class KnotPredictor:
    def __init__(self, model_path="knoten_resnet18_finetuned.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['correct', 'wrong'] 
        
        self.model = models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.class_names))
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model = self.model.to(self.device)
            self.model.eval() 
            self.is_loaded = True
        except Exception as e:
            print(f"CNN-Modell konnte nicht geladen werden: {e}")
            self.is_loaded = False
            
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_with_pipeline(self, img_bgr, p_cnn):
        if not self.is_loaded: return "Fehler", 0.0, None
        
        # Canny Pipeline Mask
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (p_cnn['blur_k'], p_cnn['blur_k']), 0)
        edges = cv2.Canny(blurred, p_cnn['canny_l'], p_cnn['canny_h'], apertureSize=3, L2gradient=True)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p_cnn['c_k'], p_cnn['c_k']))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=p_cnn['c_iter'])
        edges_dilated = cv2.dilate(edges_closed, kernel_close, iterations=p_cnn['d_iter'])
        
        mask_filled = fill_holes(edges_dilated)
        h, w = mask_filled.shape
        cv2.rectangle(mask_filled, (0, 0), (w-1, h-1), 0, thickness=p_cnn['b_thick'])
        
        mask_blurred = cv2.GaussianBlur(mask_filled, (p_cnn['s_blur'], p_cnn['s_blur']), 0)
        _, mask_smooth = cv2.threshold(mask_blurred, 127, 255, cv2.THRESH_BINARY)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p_cnn['o_k'], p_cnn['o_k']))
        mask_clean = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel_open, iterations=p_cnn['o_iter'])
        mask_main, _ = keep_largest_component(mask_clean)
        
        # find ROI
        temp = fill_holes(mask_main)
        binary = (temp > 0).astype('uint8')
        h_img, w_img = binary.shape[:2]
        k_b = int(w_img * 0.2)
        if k_b % 2 == 0: k_b += 1
        density = cv2.blur(binary.astype('float32'), (k_b, k_b))
        max_den = float(density.max())
        if max_den == 0.0: return "Kein Knoten", 0.0, None
        
        hot = np.zeros_like(binary, dtype='uint8')
        hot[density >= max_den * 0.6] = 255
        ys, xs = np.where(hot > 0)
        if len(ys) == 0: ys, xs = np.where(mask_main > 0)
        
        x0, y0 = xs.min(), ys.min()
        x1, y1 = xs.max(), ys.max()
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        ww, hh = x1 - x0, y1 - y0
        side = max(int(max(ww, hh) * 1.2), 224)
        
        x1_box = max(0, cx - side // 2)
        y1_box = max(0, cy - side // 2)
        x2_box = min(w_img, cx + side // 2)
        y2_box = min(h_img, cy + side // 2)
        
        knot_crop = img_bgr[y1_box:y2_box, x1_box:x2_box]
        knot_mask_rough = mask_main[y1_box:y2_box, x1_box:x2_box]
        
        if knot_crop.size == 0: return "Fehler", 0.0, None
        
        # GrabCut
        gc_mask = np.where(knot_mask_rough > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype('uint8')
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(knot_crop, gc_mask, None, bgdModel, fgdModel, p_cnn['gc_iter'], cv2.GC_INIT_WITH_MASK)
        except: pass
        
        refined_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        refined_mask, _ = keep_largest_component(refined_mask)
        kernel_gc_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel_gc_close, None, None, 1)

        # CNN preparation
        gray_crop = cv2.cvtColor(knot_crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_crop)
        knot_no_bg = cv2.bitwise_and(enhanced, enhanced, mask=refined_mask)
        
        h_c, w_c = knot_no_bg.shape[:2]
        max_s = max(h_c, w_c)
        top = (max_s - h_c) // 2
        bottom = max_s - h_c - top
        left = (max_s - w_c) // 2
        right = max_s - w_c - left

        padded = cv2.copyMakeBorder(knot_no_bg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        cnn_ready = cv2.resize(padded, (224, 224), interpolation=cv2.INTER_AREA)

        # prediction
        cnn_ready_rgb = cv2.cvtColor(cnn_ready, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(cnn_ready_rgb)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            max_prob, predicted_idx = torch.max(probabilities, 0)
            predicted_class = self.class_names[predicted_idx.item()]
            confidence = max_prob.item() * 100

        # debug overlay
        scale = 224.0 / knot_crop.shape[0]
        orig_resized = cv2.resize(knot_crop, (int(knot_crop.shape[1] * scale), 224))
        cnn_bgr = cnn_ready_rgb.copy()
        combined = cv2.hconcat([orig_resized, cnn_bgr])
        
        canvas_h = combined.shape[0] + 80
        canvas_w = combined.shape[1]
        debug_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        debug_img[80:, :] = combined
        
        color = (0, 255, 0) if predicted_class == 'correct' else (0, 0, 255)
        text = f"Pred: {predicted_class.upper()} ({confidence:.1f}%)"
        cv2.putText(debug_img, text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        
        return predicted_class, confidence, debug_img
    
predictor = KnotPredictor()