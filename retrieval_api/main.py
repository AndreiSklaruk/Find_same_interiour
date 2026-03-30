"""
retrieval_api/main.py
======================
FastAPI microservice wrapping image_retrieval.py search_similar() logic.
Exposes POST /search and POST /segment-doors endpoints.

Now includes MLSD Vanishing Point matching for improved camera angle detection.

Run with:
    uvicorn retrieval_api.main:app --reload --port 8000
"""

import sys
import os
import base64
import json
import gc
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, AutoModelForDepthEstimation
from pydantic import BaseModel

# MLSD
sys.path.insert(0, str(Path(__file__).parent.parent))
from mlsd.model import MobileV2_MLSD_Tiny
from mlsd.utils import pred_lines

# ---------------------------------------------------------------------------
# Paths — all relative to proj_2.1/ root
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
INDEX_JSON_PATH = BASE_DIR / "staging_index.json"
INDEX_FLOOR_MASKS = BASE_DIR / "staging_floor_masks.npy"
INDEX_CEILING_MASKS = BASE_DIR / "staging_ceiling_masks.npy"
INDEX_FURNITURE_MASKS = BASE_DIR / "staging_furniture_masks.npy"
INDEX_DEPTH_MAPS = BASE_DIR / "staging_depth_maps.npy"
INDEX_LINE_HISTOGRAMS = BASE_DIR / "staging_line_histograms.npy"
DATABASE_DIR = BASE_DIR / "database"
MLSD_WEIGHTS_PATH = BASE_DIR / "mlsd" / "weights" / "mlsd_tiny_512_fp32.pth"

# Количество бинов для гистограммы углов линий
NUM_ANGLE_BINS = 12

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Image Retrieval API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global model cache (loaded once on startup)
# ---------------------------------------------------------------------------
_processor = None
_model = None
_depth_processor = None
_depth_model = None
_mlsd_model = None
_mlsd_device = None
_device = None
_db = None
_db_floor_masks = None
_db_ceiling_masks = None
_db_furniture_masks = None
_db_depth_maps = None
_db_line_histograms = None


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@app.on_event("startup")
def load_resources():
    global _processor, _model, _depth_processor, _depth_model, _device
    global _mlsd_model, _mlsd_device
    global _db, _db_floor_masks, _db_ceiling_masks, _db_furniture_masks, _db_depth_maps
    global _db_line_histograms
    
    print("[INFO] Loading SegFormer model...")
    _device = get_device()
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    _processor = AutoImageProcessor.from_pretrained(model_name)
    _model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    _model.to(_device)
    _model.eval()
    print(f"[INFO] SegFormer loaded on {_device}")

    # Depth Anything V2 Small
    depth_model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    print(f"[INFO] Loading Depth model: {depth_model_name}")
    _depth_processor = AutoImageProcessor.from_pretrained(depth_model_name)
    _depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_name)
    _depth_model.to(_device)
    _depth_model.eval()
    print(f"[INFO] Depth model loaded on {_device}")

    # MLSD Tiny
    print("[INFO] Loading MLSD Tiny model...")
    _mlsd_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    _mlsd_model = MobileV2_MLSD_Tiny()
    _mlsd_model.load_state_dict(torch.load(MLSD_WEIGHTS_PATH, map_location=_mlsd_device), strict=True)
    _mlsd_model.to(_mlsd_device)
    _mlsd_model.eval()
    print(f"[INFO] MLSD Tiny loaded on {_mlsd_device}")

    print("[INFO] Loading index...")
    with open(INDEX_JSON_PATH, "r", encoding="utf-8") as f:
        _db = json.load(f)
    _db_floor_masks = np.load(INDEX_FLOOR_MASKS)
    _db_ceiling_masks = np.load(INDEX_CEILING_MASKS)
    _db_furniture_masks = np.load(INDEX_FURNITURE_MASKS)
    _db_depth_maps = np.load(INDEX_DEPTH_MAPS)
    
    # Загружаем гистограммы линий (если есть)
    if INDEX_LINE_HISTOGRAMS.exists():
        _db_line_histograms = np.load(INDEX_LINE_HISTOGRAMS)
        print(f"[INFO] Line histograms loaded: {_db_line_histograms.shape}")
    else:
        _db_line_histograms = None
        print("[WARN] Line histograms not found — VP similarity will be disabled")
    
    print(f"[INFO] Index loaded: {len(_db)} rooms (with depth + ceiling + MLSD VP)")


# ---------------------------------------------------------------------------
# Core analysis (adapted from image_retrieval.py)
# ---------------------------------------------------------------------------
def analyze_topology(image: Image.Image):
    """Analyze room topology from a PIL Image."""
    image = image.convert("RGB")
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)

    H, W = 128, 128
    inputs = _processor(images=image, return_tensors="pt").to(_device)
    with torch.no_grad():
        outputs = _model(**inputs)
        logits = torch.nn.functional.interpolate(
            outputs.logits, size=(H, W), mode="bilinear", align_corners=False
        )
        preds = logits.argmax(dim=1).squeeze().cpu().numpy()

    floor_mask = (preds == 3).astype(np.uint8)
    window_mask = (preds == 8).astype(np.uint8)
    door_mask = (preds == 14).astype(np.uint8)

    clean_floor = np.zeros_like(floor_mask)
    contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        cv2.fillPoly(clean_floor, [largest_cnt], 1)

    top_edge = np.full(W, H)
    for x in range(W):
        y_pts = np.where(clean_floor[:, x])[0]
        if len(y_pts) > 0:
            top_edge[x] = y_pts[0]

    peak_y = np.min(top_edge)
    peak_points = np.where(top_edge == peak_y)[0]
    corner_x = int(np.mean(peak_points))
    corner_x_ratio = corner_x / W

    flat_x = np.where(top_edge <= peak_y + 0.05 * H)[0]
    if len(flat_x) > 0 and (flat_x[-1] - flat_x[0] > 0.20 * W):
        room_type = "Frontal"
    else:
        room_type = "Corner"

    if corner_x_ratio < 0.45:
        dominant_wall = "Right"
    elif corner_x_ratio > 0.55:
        dominant_wall = "Left"
    else:
        dominant_wall = "Symmetric"

    w_left = np.sum(window_mask[:, :W // 3]) > (W * H * 0.01)
    w_center = np.sum(window_mask[:, W // 3:2 * W // 3]) > (W * H * 0.01)
    w_right = np.sum(window_mask[:, 2 * W // 3:]) > (W * H * 0.01)
    windows = [bool(w_left), bool(w_center), bool(w_right)]

    floor_ratio = float(np.sum(clean_floor) / (W * H))

    # Маска потолка
    ceiling_mask = (preds == 5).astype(np.uint8)
    clean_ceiling = np.zeros_like(ceiling_mask)
    ceil_contours, _ = cv2.findContours(ceiling_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if ceil_contours:
        largest_ceil = max(ceil_contours, key=cv2.contourArea)
        cv2.fillPoly(clean_ceiling, [largest_ceil], 1)

    small_floor = cv2.resize(clean_floor, (64, 64), interpolation=cv2.INTER_NEAREST)
    small_ceiling = cv2.resize(clean_ceiling, (64, 64), interpolation=cv2.INTER_NEAREST)
    no_go_zone = np.logical_or(window_mask, door_mask).astype(np.uint8)
    small_no_go_zone = cv2.resize(no_go_zone, (64, 64), interpolation=cv2.INTER_NEAREST)

    # Wall layout ratios (class 0 = wall in ADE20K)
    wall_mask = (preds == 0).astype(np.uint8)
    mid = W // 2
    left_half_pixels = H * mid
    right_half_pixels = H * (W - mid)
    wall_left_ratio = float(np.sum(wall_mask[:, :mid])) / left_half_pixels if left_half_pixels > 0 else 0.0
    wall_right_ratio = float(np.sum(wall_mask[:, mid:])) / right_half_pixels if right_half_pixels > 0 else 0.0

    del inputs, outputs, logits, image
    if str(_device) == "mps":
        torch.mps.empty_cache()
    gc.collect()

    # Window position: normalized X centroid (0=left, 1=right, -1=no windows)
    win_pixels = np.sum(window_mask)
    if win_pixels > 0:
        ys, xs = np.where(window_mask > 0)
        window_x_center = float(np.mean(xs)) / W
    else:
        window_x_center = -1.0

    return room_type, dominant_wall, corner_x_ratio, windows, floor_ratio, small_floor, small_ceiling, small_no_go_zone, wall_left_ratio, wall_right_ratio, preds, window_x_center


def extract_depth_map(image: Image.Image):
    """Extract depth map: 64x64 for similarity + full-res for MLSD."""
    image = image.convert("RGB")
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)

    inputs = _depth_processor(images=image, return_tensors="pt").to(_device)
    with torch.no_grad():
        outputs = _depth_model(**inputs)
        depth = outputs.predicted_depth
        
        # Full-res for MLSD
        depth_full = depth.squeeze().cpu().numpy()
        
        # 64x64 for similarity
        depth_64 = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(64, 64), mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()

    d_min, d_max = depth_64.min(), depth_64.max()
    if d_max - d_min > 1e-6:
        depth_64 = (depth_64 - d_min) / (d_max - d_min)
    else:
        depth_64 = np.zeros_like(depth_64)

    del inputs, outputs, depth
    if str(_device) == "mps":
        torch.mps.empty_cache()
    gc.collect()

    return depth_64.astype(np.float32), depth_full.astype(np.float32)


def compute_depth_similarity(depth1, depth2):
    """NCC between two depth maps. Returns -1 to 1."""
    d1 = depth1.flatten().astype(np.float64)
    d2 = depth2.flatten().astype(np.float64)
    d1 -= d1.mean()
    d2 -= d2.mean()
    norm1, norm2 = np.linalg.norm(d1), np.linalg.norm(d2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return float(np.dot(d1, d2) / (norm1 * norm2))


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# MLSD Vanishing Point functions
# ---------------------------------------------------------------------------
def find_vanishing_point(lines, img_w, img_h):
    """RANSAC-based vanishing point detection from line segments."""
    if len(lines) < 2:
        return 0.5, 0.5

    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        if length < 10:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle < 5 or angle > 175 or (85 < angle < 95):
            continue
        filtered_lines.append(line)

    if len(filtered_lines) < 2:
        return 0.5, 0.5

    filtered_lines = np.array(filtered_lines)
    x1s, y1s, x2s, y2s = filtered_lines[:, 0], filtered_lines[:, 1], filtered_lines[:, 2], filtered_lines[:, 3]
    a_arr = y2s - y1s
    b_arr = x1s - x2s
    c_arr = a_arr * x1s + b_arr * y1s

    best_vp = np.array([img_w / 2, img_h / 2])
    best_inliers = 0
    n_lines = len(filtered_lines)
    n_iterations = min(500, n_lines * (n_lines - 1) // 2)

    rng = np.random.default_rng(42)
    
    for _ in range(n_iterations):
        idx = rng.choice(n_lines, size=2, replace=False)
        i, j = idx[0], idx[1]
        det = a_arr[i] * b_arr[j] - a_arr[j] * b_arr[i]
        if abs(det) < 1e-10:
            continue
        px = (c_arr[i] * b_arr[j] - c_arr[j] * b_arr[i]) / det
        py = (a_arr[i] * c_arr[j] - a_arr[j] * c_arr[i]) / det
        if px < -img_w * 0.5 or px > img_w * 1.5 or py < -img_h * 0.5 or py > img_h * 1.5:
            continue
        dists = np.abs(a_arr * px + b_arr * py - c_arr) / (np.sqrt(a_arr ** 2 + b_arr ** 2) + 1e-10)
        inliers = np.sum(dists < 15.0)
        if inliers > best_inliers:
            best_inliers = inliers
            best_vp = np.array([px, py])

    vp_x_ratio = float(np.clip(best_vp[0] / img_w, 0, 1))
    vp_y_ratio = float(np.clip(best_vp[1] / img_h, 0, 1))
    return vp_x_ratio, vp_y_ratio


def compute_angle_histogram(lines, num_bins=NUM_ANGLE_BINS):
    """Compute normalized angle histogram from line segments."""
    if len(lines) == 0:
        return np.zeros(num_bins, dtype=np.float32)
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    angles = np.degrees(np.arctan2(dy, dx))
    angles = angles % 180
    hist, _ = np.histogram(angles, bins=num_bins, range=(0, 180))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def extract_line_features(pil_image: Image.Image, depth_map=None):
    """Extract MLSD line features from depth map (not original image!).
    Depth map removes textures/shadows, keeping only room geometry.
    Returns: vp_x, vp_y, angle_hist, x_spread, y_spread"""
    empty_result = (0.5, 0.5, np.zeros(NUM_ANGLE_BINS, dtype=np.float32), 0.25, 0.25)
    
    if depth_map is not None:
        d = depth_map.astype(np.float32)
        d = (d - d.min()) / (d.max() - d.min() + 1e-6) * 255
        depth_u8 = d.astype(np.uint8)
        mlsd_input = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2RGB)
        h, w = mlsd_input.shape[:2]
    else:
        mlsd_input = np.array(pil_image.convert("RGB"))
        h, w = mlsd_input.shape[:2]
    
    lines = pred_lines(mlsd_input, _mlsd_model, _mlsd_device,
                       input_shape=[512, 512], score_thr=0.10, dist_thr=20.0)
    
    if len(lines) == 0:
        return empty_result
    
    vp_x, vp_y = find_vanishing_point(lines, w, h)
    angle_hist = compute_angle_histogram(lines)
    
    # Room scale metric: line spread
    midpoints_x = [(l[0] + l[2]) / 2.0 / w for l in lines]
    midpoints_y = [(l[1] + l[3]) / 2.0 / h for l in lines]
    x_spread = float(np.std(midpoints_x))
    y_spread = float(np.std(midpoints_y))
    
    return vp_x, vp_y, angle_hist, x_spread, y_spread


def compute_vp_similarity(vp1_x, vp1_y, vp2_x, vp2_y, hist1, hist2):
    """Compute combined VP position + angle histogram similarity."""
    vp_dist = np.sqrt((vp1_x - vp2_x) ** 2 + (vp1_y - vp2_y) ** 2)
    vp_sim = max(0.0, 1.0 - vp_dist / 0.5)
    
    h1 = hist1.astype(np.float64)
    h2 = hist2.astype(np.float64)
    h1 -= h1.mean()
    h2 -= h2.mean()
    n1, n2 = np.linalg.norm(h1), np.linalg.norm(h2)
    if n1 < 1e-8 or n2 < 1e-8:
        hist_sim = 0.0
    else:
        hist_sim = float(np.dot(h1, h2) / (n1 * n2))
        hist_sim = max(0.0, hist_sim)
    
    return 0.7 * vp_sim + 0.3 * hist_sim


def image_to_base64(img_path: Path) -> Optional[str]:
    """Convert an image file to base64 string."""
    try:
        with open(img_path, "rb") as f:
            data = f.read()
        ext = img_path.suffix.lower().lstrip(".")
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------
class ReferenceResult(BaseModel):
    rank: int
    before_filename: str
    after_filename: str
    after_image_base64: Optional[str]
    score: float
    floor_iou: float
    ceiling_iou: float
    depth_similarity: float
    vp_similarity: float
    scale_similarity: float
    collision_percent: float
    collision_warning: bool


class SearchResponse(BaseModel):
    results: list[ReferenceResult]
    room_type: str
    dominant_wall: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "mlsd_loaded": _mlsd_model is not None}


@app.post("/search", response_model=SearchResponse)
async def search_similar(image: UploadFile = File(...), top_k: int = 5):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Read uploaded image
    contents = await image.read()
    from io import BytesIO
    pil_image = Image.open(BytesIO(contents))

    # Analyze query room topology
    q_type, q_dom_wall, _, q_wins, q_ratio, q_floor, q_ceiling, q_nogo, q_wall_left, q_wall_right, _, q_win_x = analyze_topology(pil_image)
    # Extract depth map (64x64 for similarity + full-res for MLSD)
    q_depth, q_depth_full = extract_depth_map(pil_image)
    # MLSD on depth map → clean room geometry + scale
    q_vp_x, q_vp_y, q_angle_hist, q_x_spread, q_y_spread = extract_line_features(pil_image, depth_map=q_depth_full)

    # Filter candidates by topology match (with progressive relaxation)
    # Step 1: Strict match (room_type + dominant_wall)
    candidates = []
    for filepath, data in _db.items():
        if data["room_type"] != q_type:
            continue
        if data["dominant_wall"] != q_dom_wall:
            continue
        candidates.append((filepath, data))

    # Step 2: If too few — relax dominant_wall constraint
    if len(candidates) < top_k:
        candidates = []
        for filepath, data in _db.items():
            if data["room_type"] != q_type:
                continue
            candidates.append((filepath, data))

    # Step 3: If still too few — use ALL candidates (let scoring decide)
    if len(candidates) < top_k:
        candidates = [(fp, d) for fp, d in _db.items()]

    # Score candidates
    results = []
    total_nogo_pixels = np.sum(q_nogo)

    for filepath, data in candidates:
        idx = data["index"]
        floor_iou = compute_iou(q_floor, _db_floor_masks[idx])
        ceiling_iou = compute_iou(q_ceiling, _db_ceiling_masks[idx])
        candidate_furniture = _db_furniture_masks[idx]
        blocked_pixels = np.logical_and(q_nogo, candidate_furniture).sum()
        collision_percent = float(blocked_pixels / total_nogo_pixels) if total_nogo_pixels > 0 else 0.0
        depth_sim = compute_depth_similarity(q_depth, _db_depth_maps[idx])
        area_penalty = abs(data["floor_ratio"] - q_ratio)

        # wall_sim removed — camera angle fully covered by vp_sim (VP is more precise)

        # Vanishing Point similarity (NEW!)
        c_vp_x = data.get("vp_x", 0.5)
        c_vp_y = data.get("vp_y", 0.5)
        if _db_line_histograms is not None:
            c_hist = _db_line_histograms[idx]
        else:
            c_hist = np.zeros(NUM_ANGLE_BINS, dtype=np.float32)
        vp_sim = compute_vp_similarity(q_vp_x, q_vp_y, c_vp_x, c_vp_y, q_angle_hist, c_hist)

        # Room scale similarity (line spread comparison)
        c_x_spread = data.get("x_spread", 0.25)
        c_y_spread = data.get("y_spread", 0.25)
        scale_sim = 1.0 - (abs(q_x_spread - c_x_spread) + abs(q_y_spread - c_y_spread))
        scale_sim = max(0.0, scale_sim)  # clamp to [0, 1]

        # Window position similarity
        c_win_x = data.get("window_x_center", -1.0)
        if q_win_x >= 0 and c_win_x >= 0:
            # Both have windows → compare positions (0=identical, 1=opposite sides)
            win_pos_sim = 1.0 - abs(q_win_x - c_win_x)
        else:
            # One or both have no windows → neutral
            win_pos_sim = 0.5

        final_score = (floor_iou * 0.5
                      + (ceiling_iou * 0.3) 
                      + (depth_sim * 0.5) 
                      + (vp_sim * 0.9)
                      + (scale_sim * 0.4)
                      + (win_pos_sim * 0.5)
                      - (area_penalty * 0.5) 
                      - (collision_percent * 2.0))
        results.append((final_score, filepath, data, floor_iou, ceiling_iou, collision_percent, depth_sim, vp_sim, scale_sim, win_pos_sim))

    results.sort(key=lambda x: x[0], reverse=True)

    # Build response: only keep candidates with a valid _after image
    response_results = []
    rank = 1
    
    for (score, filepath, data, iou, c_iou, collision, d_sim, vp_s, sc_sim, wp_sim) in results:
        if len(response_results) >= top_k:
            break

        # Skip candidates below minimum similarity threshold
        if score < 0.5:
            break  # Results are sorted, so all remaining will be below threshold too

        before_path = Path(filepath)
        if not before_path.is_absolute():
            before_path = BASE_DIR / filepath

        base_name = before_path.stem
        ext = before_path.suffix

        # 1. Try standard pattern: image_1_after.jpg
        after_name = f"{base_name}_after{ext}"
        after_path = before_path.with_name(after_name)

        if not after_path.exists():
            # 2. Try other extensions
            found = False
            for cand_ext in [".jpg", ".jpeg", ".png", ".JPG"]:
                candidate = before_path.with_name(f"{base_name}_after{cand_ext}")
                if candidate.exists():
                    after_path = candidate
                    found = True
                    break
            
            # If still not found, this candidate is useless as a reference -> skip it entirely
            if not found:
                continue

        after_b64 = image_to_base64(after_path)
        if not after_b64:
            continue

        response_results.append(ReferenceResult(
            rank=rank,
            before_filename=before_path.name,
            after_filename=after_path.name,
            after_image_base64=after_b64,
            score=round(score, 4),
            floor_iou=round(iou, 4),
            ceiling_iou=round(c_iou, 4),
            depth_similarity=round(d_sim, 4),
            vp_similarity=round(vp_s, 4),
            scale_similarity=round(sc_sim, 4),
            collision_percent=round(collision, 4),
            collision_warning=collision > 0.05,
        ))
        rank += 1

    return SearchResponse(
        results=response_results,
        room_type=q_type,
        dominant_wall=q_dom_wall,
    )


# ---------------------------------------------------------------------------
@app.post("/segment-doors")
async def segment_doors(image: UploadFile = File(...)):
    """
    Accept an image, run SegFormer segmentation, and return a base64 PNG mask
    where doors (class 14) and windows (class 8) are white on black background.
    The mask is returned at the original image resolution.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    contents = await image.read()
    from io import BytesIO
    pil_image = Image.open(BytesIO(contents)).convert("RGB")
    orig_w, orig_h = pil_image.size

    # Resize for inference
    resized = pil_image.copy()
    resized.thumbnail((512, 512), Image.Resampling.LANCZOS)

    inputs = _processor(images=resized, return_tensors="pt").to(_device)
    with torch.no_grad():
        outputs = _model(**inputs)
        logits = torch.nn.functional.interpolate(
            outputs.logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        preds = logits.argmax(dim=1).squeeze().cpu().numpy()

    # Build binary mask: doors (14) + windows (8)
    door_mask = (preds == 14).astype(np.uint8)
    window_mask = (preds == 8).astype(np.uint8)
    combined = np.logical_or(door_mask, window_mask).astype(np.uint8) * 255

    # Dilate slightly to cover edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    combined = cv2.dilate(combined, kernel, iterations=1)

    # Encode as PNG base64
    _, png_buf = cv2.imencode(".png", combined)
    mask_b64 = base64.b64encode(png_buf.tobytes()).decode("utf-8")

    del inputs, outputs, logits, preds
    if str(_device) == "mps":
        torch.mps.empty_cache()
    gc.collect()

    has_doors = bool(np.any(door_mask))
    has_windows = bool(np.any(window_mask))

    return {
        "mask_base64": mask_b64,
        "has_doors": has_doors,
        "has_windows": has_windows,
        "width": orig_w,
        "height": orig_h,
    }
