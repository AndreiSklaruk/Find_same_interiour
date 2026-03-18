"""
retrieval_api/main.py
======================
FastAPI microservice wrapping image_retrieval.py search_similar() logic.
Exposes a single POST /search endpoint.

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

# ---------------------------------------------------------------------------
# Paths — all relative to proj_2.1/ root
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
INDEX_JSON_PATH = BASE_DIR / "staging_index.json"
INDEX_FLOOR_MASKS = BASE_DIR / "staging_floor_masks.npy"
INDEX_CEILING_MASKS = BASE_DIR / "staging_ceiling_masks.npy"
INDEX_FURNITURE_MASKS = BASE_DIR / "staging_furniture_masks.npy"
INDEX_DEPTH_MAPS = BASE_DIR / "staging_depth_maps.npy"
DATABASE_DIR = BASE_DIR / "database"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Image Retrieval API", version="1.0.0")

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
_device = None
_db = None
_db_floor_masks = None
_db_ceiling_masks = None
_db_furniture_masks = None
_db_depth_maps = None


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@app.on_event("startup")
def load_resources():
    global _processor, _model, _depth_processor, _depth_model, _device
    global _db, _db_floor_masks, _db_ceiling_masks, _db_furniture_masks, _db_depth_maps
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

    print("[INFO] Loading index...")
    with open(INDEX_JSON_PATH, "r", encoding="utf-8") as f:
        _db = json.load(f)
    _db_floor_masks = np.load(INDEX_FLOOR_MASKS)
    _db_ceiling_masks = np.load(INDEX_CEILING_MASKS)
    _db_furniture_masks = np.load(INDEX_FURNITURE_MASKS)
    _db_depth_maps = np.load(INDEX_DEPTH_MAPS)
    print(f"[INFO] Index loaded: {len(_db)} rooms (with depth + ceiling maps)")


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

    del inputs, outputs, logits, preds
    if str(_device) == "mps":
        torch.mps.empty_cache()
    gc.collect()

    return room_type, dominant_wall, corner_x_ratio, windows, floor_ratio, small_floor, small_ceiling, small_no_go_zone


def extract_depth_map(image: Image.Image):
    """Extract normalized 64x64 depth map from a PIL Image."""
    image = image.convert("RGB")
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)

    inputs = _depth_processor(images=image, return_tensors="pt").to(_device)
    with torch.no_grad():
        outputs = _depth_model(**inputs)
        depth = outputs.predicted_depth
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

    return depth_64.astype(np.float32)


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
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/search", response_model=SearchResponse)
async def search_similar(image: UploadFile = File(...), top_k: int = 5):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Read uploaded image
    contents = await image.read()
    from io import BytesIO
    pil_image = Image.open(BytesIO(contents))

    # Analyze query room topology
    q_type, q_dom_wall, _, q_wins, q_ratio, q_floor, q_ceiling, q_nogo = analyze_topology(pil_image)
    # Extract query depth map
    q_depth = extract_depth_map(pil_image)

    # Filter candidates by topology match
    candidates = []
    for filepath, data in _db.items():
        if data["room_type"] != q_type:
            continue
        if data["dominant_wall"] != q_dom_wall:
            continue
        candidates.append((filepath, data))

    # If no exact match — relax dominant_wall constraint
    if not candidates:
        for filepath, data in _db.items():
            if data["room_type"] != q_type:
                continue
            candidates.append((filepath, data))

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
        final_score = floor_iou + (ceiling_iou * 0.3) + (depth_sim * 0.3) - (area_penalty * 0.5) - (collision_percent * 2.0)
        results.append((final_score, filepath, data, floor_iou, ceiling_iou, collision_percent, depth_sim))

    results.sort(key=lambda x: x[0], reverse=True)

    # Build response: only keep candidates with a valid _after image
    response_results = []
    rank = 1
    
    for (score, filepath, data, iou, c_iou, collision, d_sim) in results:
        if len(response_results) >= top_k:
            break

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
            collision_percent=round(collision, 4),
            collision_warning=collision > 0.05,
        ))
        rank += 1

    return SearchResponse(
        results=response_results,
        room_type=q_type,
        dominant_wall=q_dom_wall,
    )
