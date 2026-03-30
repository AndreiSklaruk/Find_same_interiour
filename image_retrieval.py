# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pillow",
#     "torch",
#     "torchvision",
#     "tqdm",
#     "transformers",
#     "opencv-python",
# ]
# ///

"""
image_retrieval.py
==================
ПОИСК ПО ТОПОЛОГИИ + ГЛУБИНА + ПРОВЕРКА КОЛЛИЗИЙ МЕБЕЛИ + MLSD VANISHING POINT.

Алгоритм:
  1. Извлекает Топологию (пустые углы и стены) из файлов before.
  2. Извлекает Depth-карту (Depth Anything V2) из файлов before.
  3. Извлекает "Отпечаток Мебели" из файлов _after.
  4. Извлекает линейные сегменты (MLSD) → Vanishing Point + гистограмма углов.
  5. При поиске сравнивает глубину, VP и накладывает мебель кандидата на окна запроса.
  6. Если мебель перекрывает свет/окна - штрафует и бракует вариант.
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import json
import gc
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, AutoModelForDepthEstimation

# MLSD
from mlsd.model import MobileV2_MLSD_Tiny
from mlsd.utils import pred_lines


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
INDEX_JSON_PATH = "staging_index.json"
INDEX_FLOOR_MASKS = "staging_floor_masks.npy"
INDEX_CEILING_MASKS = "staging_ceiling_masks.npy"
INDEX_FURNITURE_MASKS = "staging_furniture_masks.npy"
INDEX_DEPTH_MAPS = "staging_depth_maps.npy"
INDEX_LINE_HISTOGRAMS = "staging_line_histograms.npy"

MLSD_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "mlsd", "weights", "mlsd_tiny_512_fp32.pth")

# Количество бинов для гистограммы углов линий
NUM_ANGLE_BINS = 12


# ---------------------------------------------------------------------------
# 1. Загрузка моделей
# ---------------------------------------------------------------------------
def load_models():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"[INFO] Используемое устройство: {device}")
    
    # SegFormer для сегментации
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    processor = AutoImageProcessor.from_pretrained(model_name)
    seg_model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    seg_model.to(device)
    seg_model.eval()

    # Depth Anything V2 Small для карт глубины
    depth_model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    print(f"[INFO] Загрузка Depth-модели: {depth_model_name}")
    depth_processor = AutoImageProcessor.from_pretrained(depth_model_name)
    depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_name)
    depth_model.to(device)
    depth_model.eval()

    # MLSD Tiny для детекции линий
    print(f"[INFO] Загрузка MLSD Tiny модели")
    mlsd_model = MobileV2_MLSD_Tiny()
    # MLSD работает только на CPU/CUDA (не MPS — из-за custom ops)
    mlsd_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mlsd_model.load_state_dict(torch.load(MLSD_WEIGHTS_PATH, map_location=mlsd_device), strict=True)
    mlsd_model.to(mlsd_device)
    mlsd_model.eval()
    print(f"[INFO] MLSD Tiny загружен на {mlsd_device}")

    return processor, seg_model, depth_processor, depth_model, mlsd_model, mlsd_device, device


# ---------------------------------------------------------------------------
# 2. КАРТА ГЛУБИНЫ
# ---------------------------------------------------------------------------
def extract_depth_map(image_path_or_pil, depth_processor, depth_model, device):
    """Извлекает depth-карту: 64×64 для сравнения + полноразмерную для MLSD."""
    if isinstance(image_path_or_pil, (str, Path)):
        image = Image.open(image_path_or_pil).convert("RGB")
    else:
        image = image_path_or_pil.convert("RGB")
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)

    inputs = depth_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = depth_model(**inputs)
        depth = outputs.predicted_depth  # (1, H, W)
        
        # Полноразмерная depth для MLSD
        depth_full = depth.squeeze().cpu().numpy()
        
        # Resize до 64×64 для similarity matching
        depth_64 = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(64, 64), mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()

    # Нормализация 64x64 в [0, 1]
    d_min, d_max = depth_64.min(), depth_64.max()
    if d_max - d_min > 1e-6:
        depth_64 = (depth_64 - d_min) / (d_max - d_min)
    else:
        depth_64 = np.zeros_like(depth_64)

    del inputs, outputs, depth
    if str(device) == 'mps': torch.mps.empty_cache()
    gc.collect()

    return depth_64.astype(np.float32), depth_full.astype(np.float32)


def compute_depth_similarity(depth1, depth2):
    """Normalized Cross-Correlation (NCC) между двумя depth-картами.
    Возвращает значение от -1 до 1 (1 = идеальное совпадение)."""
    d1 = depth1.flatten().astype(np.float64)
    d2 = depth2.flatten().astype(np.float64)
    d1 -= d1.mean()
    d2 -= d2.mean()
    norm1, norm2 = np.linalg.norm(d1), np.linalg.norm(d2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return float(np.dot(d1, d2) / (norm1 * norm2))


# ---------------------------------------------------------------------------
# 3. MLSD VANISHING POINT + ГИСТОГРАММА УГЛОВ
# ---------------------------------------------------------------------------
def find_vanishing_point(lines, img_w, img_h):
    """
    Находит точку схода (vanishing point) методом RANSAC.
    
    Args:
        lines: numpy array (N, 4) — [x1, y1, x2, y2]
        img_w, img_h: размеры изображения
    
    Returns:
        (vp_x_ratio, vp_y_ratio): нормализованные координаты VP (0-1)
        Если VP не найден, возвращает (0.5, 0.5) — центр изображения.
    """
    if len(lines) < 2:
        return 0.5, 0.5

    # Фильтруем почти горизонтальные и вертикальные линии (они не сходятся)
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        if length < 10:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        # Пропускаем строго горизонтальные (±5°) и вертикальные (85-95°)
        if angle < 5 or angle > 175 or (85 < angle < 95):
            continue
        filtered_lines.append(line)

    if len(filtered_lines) < 2:
        return 0.5, 0.5

    filtered_lines = np.array(filtered_lines)

    # Переводим линии в форму ax + by = c
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
        # Выбираем две случайные линии
        idx = rng.choice(n_lines, size=2, replace=False)
        i, j = idx[0], idx[1]

        # Вычисляем пересечение
        det = a_arr[i] * b_arr[j] - a_arr[j] * b_arr[i]
        if abs(det) < 1e-10:
            continue

        px = (c_arr[i] * b_arr[j] - c_arr[j] * b_arr[i]) / det
        py = (a_arr[i] * c_arr[j] - a_arr[j] * c_arr[i]) / det

        # VP должен быть в разумных пределах (±50% от размера изображения)
        if px < -img_w * 0.5 or px > img_w * 1.5 or py < -img_h * 0.5 or py > img_h * 1.5:
            continue

        # Считаем inliers: линии, проходящие через эту точку
        # Расстояние от точки (px, py) до каждой линии
        dists = np.abs(a_arr * px + b_arr * py - c_arr) / (np.sqrt(a_arr ** 2 + b_arr ** 2) + 1e-10)
        inliers = np.sum(dists < 15.0)  # порог: 15 пикселей

        if inliers > best_inliers:
            best_inliers = inliers
            best_vp = np.array([px, py])

    # Нормализуем
    vp_x_ratio = float(np.clip(best_vp[0] / img_w, 0, 1))
    vp_y_ratio = float(np.clip(best_vp[1] / img_h, 0, 1))
    
    return vp_x_ratio, vp_y_ratio


def compute_angle_histogram(lines, num_bins=NUM_ANGLE_BINS):
    """
    Вычисляет гистограмму углов линий.
    
    Args:
        lines: numpy array (N, 4) — [x1, y1, x2, y2]
        num_bins: количество бинов (0° - 180°)
    
    Returns:
        histogram: numpy array (num_bins,) — нормализованная гистограмма
    """
    if len(lines) == 0:
        return np.zeros(num_bins, dtype=np.float32)

    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    angles = np.degrees(np.arctan2(dy, dx))
    angles = angles % 180  # Приводим к [0, 180)

    hist, _ = np.histogram(angles, bins=num_bins, range=(0, 180))
    hist = hist.astype(np.float32)
    
    # Нормализация
    total = hist.sum()
    if total > 0:
        hist /= total

    return hist


def filter_structural_lines(lines, seg_preds, img_h, img_w):
    """
    Оставляет ТОЛЬКО линии на структурных элементах комнаты:
    стены (class 0), потолок (5), окна (8), двери (14).
    
    Убирает тени на полу, паркет, мебель и прочий шум.
    Проверяет середину + оба конца каждой линии.
    
    Args:
        lines: numpy array (N, 4) — [x1, y1, x2, y2]
        seg_preds: numpy array (H, W) — карта сегментации ADE20K
        img_h, img_w: размеры оригинального изображения
    """
    if len(lines) == 0 or seg_preds is None:
        return lines
    
    # Структурные классы ADE20K: стены, потолок, окна, двери
    STRUCTURAL_CLASSES = {0, 5, 8, 14}
    
    mask_h, mask_w = seg_preds.shape
    filtered = []
    
    for line in lines:
        x1, y1, x2, y2 = line
        
        # Проверяем три точки: начало, середину, конец
        points = [
            (x1, y1),
            ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            (x2, y2)
        ]
        
        structural_count = 0
        for px, py in points:
            mx = int(np.clip(px / img_w * mask_w, 0, mask_w - 1))
            my = int(np.clip(py / img_h * mask_h, 0, mask_h - 1))
            if seg_preds[my, mx] in STRUCTURAL_CLASSES:
                structural_count += 1
        
        # Линия структурная если хотя бы 2 из 3 точек на стенах/потолке/окнах
        if structural_count >= 2:
            filtered.append(line)
    
    return np.array(filtered).reshape(-1, 4) if filtered else np.array([]).reshape(0, 4)


def extract_line_features(image_path_or_pil, mlsd_model, mlsd_device, depth_map=None):
    """
    Извлекает линейные фичи: VP + гистограмму углов + метрику масштаба.
    
    КЛЮЧЕВАЯ ИДЕЯ: MLSD запускается на depth-карте, а НЕ на оригинальной фотографии.
    Depth-карта убирает текстуры (паркет, тени, отражения) и оставляет только
    чистую геометрию комнаты — стыки стен, пол, потолок, окна.
    
    Args:
        depth_map: numpy array (H, W) — depth map из Depth Anything V2 (float).
                   Если передан, MLSD работает по нему. Иначе — по оригиналу.
    
    Returns:
        vp_x, vp_y: нормализованные координаты vanishing point
        angle_hist: numpy array (NUM_ANGLE_BINS,) — гистограмма углов
        x_spread: float — std X-координат середин линий (метрика масштаба)
        y_spread: float — std Y-координат середин линий
    """
    empty_result = (0.5, 0.5, np.zeros(NUM_ANGLE_BINS, dtype=np.float32), 0.25, 0.25)
    
    if depth_map is not None:
        # Нормализуем depth в 0-255 и конвертируем в RGB для MLSD
        d = depth_map.astype(np.float32)
        d = (d - d.min()) / (d.max() - d.min() + 1e-6) * 255
        depth_u8 = d.astype(np.uint8)
        mlsd_input = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2RGB)
        h, w = mlsd_input.shape[:2]
    else:
        # Fallback: работаем по оригинальному изображению
        if isinstance(image_path_or_pil, (str, Path)):
            img = cv2.imread(str(image_path_or_pil))
            if img is None:
                return empty_result
            mlsd_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            mlsd_input = np.array(image_path_or_pil.convert("RGB"))
        h, w = mlsd_input.shape[:2]
    
    # MLSD inference на depth-карте
    lines = pred_lines(mlsd_input, mlsd_model, mlsd_device, 
                       input_shape=[512, 512], score_thr=0.10, dist_thr=20.0)
    
    if len(lines) == 0:
        return empty_result

    # Vanishing point
    vp_x, vp_y = find_vanishing_point(lines, w, h)
    
    # Гистограмма углов
    angle_hist = compute_angle_histogram(lines)
    
    # Метрика масштаба: разброс линий
    # Маленькая комната → линии сконцентрированы (spread ~0.15-0.20)
    # Большая комната → линии разбросаны (spread ~0.30-0.40)
    midpoints_x = [(l[0] + l[2]) / 2.0 / w for l in lines]
    midpoints_y = [(l[1] + l[3]) / 2.0 / h for l in lines]
    x_spread = float(np.std(midpoints_x))
    y_spread = float(np.std(midpoints_y))
    
    return vp_x, vp_y, angle_hist, x_spread, y_spread


def compute_vp_similarity(vp1_x, vp1_y, vp2_x, vp2_y, hist1, hist2):
    """
    Вычисляет сходство vanishing points + гистограмм углов.
    
    Returns:
        similarity: float от 0 до 1 (1 = идеальное совпадение)
    """
    # 1. Евклидово расстояние между VP (в нормализованных координатах)
    vp_dist = np.sqrt((vp1_x - vp2_x) ** 2 + (vp1_y - vp2_y) ** 2)
    # Максимально возможное расстояние в нормализованных координатах = sqrt(2) ≈ 1.414
    vp_sim = max(0.0, 1.0 - vp_dist / 0.5)  # крутой спад: при dist=0.5 уже 0
    
    # 2. Корреляция гистограмм углов
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
    
    # Комбинируем: VP важнее (70%), гистограмма дополняет (30%)
    return 0.7 * vp_sim + 0.3 * hist_sim


# ---------------------------------------------------------------------------
# 4. АНАЛИЗ ГЕОМЕТРИИ И ОТПЕЧАТКОВ МЕБЕЛИ
# ---------------------------------------------------------------------------
def analyze_topology(image_path: str | Path, processor, seg_model, device):
    """Анализирует ПУСТУЮ комнату (каркас и запретные зоны)."""
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    
    H, W = 128, 128
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = seg_model(**inputs)
        logits = torch.nn.functional.interpolate(outputs.logits, size=(H, W), mode="bilinear", align_corners=False)
        preds = logits.argmax(dim=1).squeeze().cpu().numpy()

    floor_mask = (preds == 3).astype(np.uint8)
    window_mask = (preds == 8).astype(np.uint8)
    door_mask = (preds == 14).astype(np.uint8)

    clean_floor = np.zeros_like(floor_mask)
    contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        cv2.fillPoly(clean_floor, [largest_cnt], 1)

    # Правило 1 и 2: Ищем "Дальний Угол" и Тип комнаты
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

    w_left   = np.sum(window_mask[:, :W//3]) > (W * H * 0.01)
    w_center = np.sum(window_mask[:, W//3:2*W//3]) > (W * H * 0.01)
    w_right  = np.sum(window_mask[:, 2*W//3:]) > (W * H * 0.01)
    windows = [bool(w_left), bool(w_center), bool(w_right)]

    floor_ratio = float(np.sum(clean_floor) / (W * H))
    
    # Маска потолка
    ceiling_mask = (preds == 5).astype(np.uint8)
    clean_ceiling = np.zeros_like(ceiling_mask)
    ceil_contours, _ = cv2.findContours(ceiling_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if ceil_contours:
        largest_ceil = max(ceil_contours, key=cv2.contourArea)
        cv2.fillPoly(clean_ceiling, [largest_ceil], 1)
    
    # Собираем маски (64x64)
    small_floor = cv2.resize(clean_floor, (64, 64), interpolation=cv2.INTER_NEAREST)
    small_ceiling = cv2.resize(clean_ceiling, (64, 64), interpolation=cv2.INTER_NEAREST)
    
    # "Запретная зона" - окна и двери, которые нельзя перекрывать мебелью
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
    if str(device) == 'mps': torch.mps.empty_cache()
    gc.collect()
    # Позиция окон: нормализованный X центра масс (0=лево, 1=право, -1=нет окон)
    win_pixels = np.sum(window_mask)
    if win_pixels > 0:
        ys, xs = np.where(window_mask > 0)
        window_x_center = float(np.mean(xs)) / W  # 0.0 = left edge, 1.0 = right edge
    else:
        window_x_center = -1.0  # Нет окон

    return room_type, dominant_wall, corner_x_ratio, windows, floor_ratio, small_floor, small_ceiling, small_no_go_zone, wall_left_ratio, wall_right_ratio, preds, window_x_center


def extract_furniture_footprint(after_image_path: Path, processor, seg_model, device):
    """Вырезает силуэт мебели из картинки _after."""
    image = Image.open(after_image_path).convert("RGB")
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = seg_model(**inputs)
        logits = torch.nn.functional.interpolate(outputs.logits, size=(64, 64), mode="bilinear", align_corners=False)
        preds = logits.argmax(dim=1).squeeze().cpu().numpy()

    # Гениальный хак: Мебель - это всё, что НЕ является структурными элементами комнаты
    # 0:wall, 3:floor, 5:ceiling, 8:window, 14:door
    furniture_mask = ((preds != 0) & (preds != 3) & (preds != 5) & (preds != 8) & (preds != 14)).astype(np.uint8)

    del inputs, outputs, logits, preds, image
    if str(device) == 'mps': torch.mps.empty_cache()
    gc.collect()

    return furniture_mask


# ---------------------------------------------------------------------------
# 5. ИНДЕКСАЦИЯ БАЗЫ (BEFORE + AFTER)
# ---------------------------------------------------------------------------
def build_index(database_dir, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    database_dir = Path(database_dir)
    image_paths = sorted([p for p in database_dir.rglob("*") if p.suffix.lower() in extensions])

    if not image_paths:
        raise ValueError("Изображений не найдено.")

    processor, seg_model, depth_processor, depth_model, mlsd_model, mlsd_device, device = load_models()
    logical_db = {}
    floor_masks = []
    ceiling_masks = []
    furniture_masks = []
    depth_maps = []
    line_histograms = []

    for img_path in tqdm(image_paths, desc="Индексация", unit="img"):
        # Мы индексируем только пустые болванки, но подглядываем в их _after версии
        if "_after" in img_path.stem.lower():
            continue

        try:
            # 1. Анализ пустой комнаты
            r_type, dom_wall, corner_x, wins, ratio, floor_m, ceil_m, _, wl_ratio, wr_ratio, _, win_x = analyze_topology(img_path, processor, seg_model, device)
            
            # 2. Извлечение карты глубины (64x64 для similarity + полноразмер для MLSD)
            depth_m, depth_full = extract_depth_map(img_path, depth_processor, depth_model, device)
            
            # 3. Поиск готового дизайна (_after)
            after_path = img_path.with_name(f"{img_path.stem}_after{img_path.suffix}")
            if not after_path.exists():
                possible_afters = list(img_path.parent.glob(f"{img_path.stem}_after.*"))
                if possible_afters:
                    after_path = possible_afters[0]
            
            # 4. Извлечение мебели
            if after_path.exists():
                furn_m = extract_furniture_footprint(after_path, processor, seg_model, device)
            else:
                furn_m = np.zeros((64, 64), dtype=np.uint8) # Мебели нет

            # 5. MLSD на depth-карте → VP + гистограмма + масштаб (чистая геометрия)
            vp_x, vp_y, angle_hist, x_spread, y_spread = extract_line_features(img_path, mlsd_model, mlsd_device, depth_map=depth_full)

            # Сохраняем в базу
            idx = len(floor_masks)
            logical_db[str(img_path)] = {
                "index": idx,
                "room_type": r_type,
                "dominant_wall": dom_wall,
                "corner_x": corner_x,
                "windows": wins,
                "floor_ratio": ratio,
                "wall_left_ratio": round(wl_ratio, 4),
                "wall_right_ratio": round(wr_ratio, 4),
                "vp_x": round(vp_x, 4),
                "vp_y": round(vp_y, 4),
                "x_spread": round(x_spread, 4),
                "y_spread": round(y_spread, 4),
                "window_x_center": round(win_x, 4)
            }
            floor_masks.append(floor_m)
            ceiling_masks.append(ceil_m)
            furniture_masks.append(furn_m)
            depth_maps.append(depth_m)
            line_histograms.append(angle_hist)

        except Exception as e:
            print(f"[WARN] Пропущен {img_path.name}: {e}")

    with open(INDEX_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(logical_db, f, ensure_ascii=False, indent=2)
    
    np.save(INDEX_FLOOR_MASKS, np.stack(floor_masks))
    np.save(INDEX_CEILING_MASKS, np.stack(ceiling_masks))
    np.save(INDEX_FURNITURE_MASKS, np.stack(furniture_masks))
    np.save(INDEX_DEPTH_MAPS, np.stack(depth_maps))
    np.save(INDEX_LINE_HISTOGRAMS, np.stack(line_histograms))
    print(f"[OK] База стейджинга сохранена! ({len(logical_db)} комнат с мебелью + depth + ceiling + MLSD VP)")


# ---------------------------------------------------------------------------
# 6. ПОИСК С ПРОВЕРКОЙ КОЛЛИЗИЙ + ГЛУБИНА + VANISHING POINT
# ---------------------------------------------------------------------------
def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0

def search_similar(query_image_path, top_k=5):
    if not os.path.exists(INDEX_JSON_PATH):
        raise FileNotFoundError("Индекс не найден. Запустите index.")

    with open(INDEX_JSON_PATH, "r") as f:
        db = json.load(f)
    db_floor_masks = np.load(INDEX_FLOOR_MASKS)
    db_ceiling_masks = np.load(INDEX_CEILING_MASKS)
    db_furniture_masks = np.load(INDEX_FURNITURE_MASKS)
    db_depth_maps = np.load(INDEX_DEPTH_MAPS)
    
    # Загружаем гистограммы линий (если есть)
    if os.path.exists(INDEX_LINE_HISTOGRAMS):
        db_line_histograms = np.load(INDEX_LINE_HISTOGRAMS)
    else:
        db_line_histograms = None

    processor, seg_model, depth_processor, depth_model, mlsd_model, mlsd_device, device = load_models()

    print(f"\n[INFO] АНАЛИЗ ЗАПРОСА: {Path(query_image_path).name}")
    # Для запроса нам важна No-Go Zone (Окна и Двери)
    q_type, q_dom_wall, _, q_wins, q_ratio, q_floor, q_ceiling, q_nogo, q_wall_left, q_wall_right, _, q_win_x = analyze_topology(query_image_path, processor, seg_model, device)
    # Извлекаем depth-карту запроса (64x64 для similarity + полноразмер для MLSD)
    q_depth, q_depth_full = extract_depth_map(query_image_path, depth_processor, depth_model, device)
    # MLSD на depth-карте → чистая геометрия + масштаб
    q_vp_x, q_vp_y, q_angle_hist, q_x_spread, q_y_spread = extract_line_features(query_image_path, mlsd_model, mlsd_device, depth_map=q_depth_full)
    
    print(f" ├─ Тип комнаты: {'Фронтальная' if q_type == 'Frontal' else 'Угловая'}")
    print(f" ├─ Доминирует: {'Стена 1' if q_dom_wall == 'Left' else 'Стена 2' if q_dom_wall == 'Right' else 'Симметрия'}")
    print(f" ├─ Depth-карта: извлечена (64×64)")
    print(f" ├─ Потолок: {np.sum(q_ceiling)} пикселей")
    print(f" ├─ Vanishing Point: ({q_vp_x:.3f}, {q_vp_y:.3f})")
    print(f" └─ Запретных зон для мебели (окна/двери): {np.sum(q_nogo)} пикселей\n")

    # =========================================================
    # ШАГ 1: Жесткая Топология
    # =========================================================
    candidates = []
    for filepath, data in db.items():
        if data["room_type"] != q_type: continue
        if data["dominant_wall"] != q_dom_wall: continue
        candidates.append(filepath)

    print(f"[ФИЛЬТР] По совпадению каркаса (Стены 1, 2, 3) отобрано: {len(candidates)} комнат.")

    # =========================================================
    # ШАГ 2: Виртуальная примерка мебели + Глубина + VP
    # =========================================================
    results = []
    total_nogo_pixels = np.sum(q_nogo)

    for filepath in candidates:
        data = db[filepath]
        idx = data["index"]
        
        # 1. Насколько хорошо совпал пол
        floor_iou = compute_iou(q_floor, db_floor_masks[idx])
        
        # 2. Насколько хорошо совпал потолок
        ceiling_iou = compute_iou(q_ceiling, db_ceiling_masks[idx])
        
        # 3. ПРОВЕРКА КОЛЛИЗИЙ (Накладываем чужую мебель на наши окна)
        candidate_furniture = db_furniture_masks[idx]
        blocked_pixels = np.logical_and(q_nogo, candidate_furniture).sum()
        
        # Какой процент окна перекрыт?
        collision_percent = (blocked_pixels / total_nogo_pixels) if total_nogo_pixels > 0 else 0.0
        
        # 4. Совпадение глубины (NCC)
        depth_sim = compute_depth_similarity(q_depth, db_depth_maps[idx])
        
        # 5. Wall layout similarity
        c_wall_left = data.get("wall_left_ratio", 0.0)
        c_wall_right = data.get("wall_right_ratio", 0.0)
        wall_sim = 1.0 - (abs(q_wall_left - c_wall_left) + abs(q_wall_right - c_wall_right)) / 2.0

        # 6. Vanishing Point similarity (NEW!)
        c_vp_x = data.get("vp_x", 0.5)
        c_vp_y = data.get("vp_y", 0.5)
        if db_line_histograms is not None:
            c_hist = db_line_histograms[idx]
        else:
            c_hist = np.zeros(NUM_ANGLE_BINS, dtype=np.float32)
        vp_sim = compute_vp_similarity(q_vp_x, q_vp_y, c_vp_x, c_vp_y, q_angle_hist, c_hist)

        # 7. Финальный балл
        area_penalty = abs(data["floor_ratio"] - q_ratio)

        final_score = (floor_iou 
                      + (ceiling_iou * 0.3) 
                      + (depth_sim * 0.3) 
                      + (wall_sim * 0.4) 
                      + (vp_sim * 0.5)       # ← NEW: Vanishing Point
                      - (area_penalty * 0.5) 
                      - (collision_percent * 2.0))
        
        results.append((final_score, filepath, floor_iou, ceiling_iou, collision_percent, depth_sim, vp_sim))

    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:top_k]

    print(f"\n{'='*140}")
    print(f" ТОП-{top_k} ИДЕАЛЬНЫХ СОВПАДЕНИЙ С УЧЕТОМ МЕБЕЛИ, ГЛУБИНЫ, ПОТОЛКА И РАКУРСА (VP):")
    print(f"{'='*140}")
    
    for rank, (score, filepath, iou, c_iou, collision, d_sim, vp_s) in enumerate(top_results, start=1):
        before_path = Path(filepath)
        after_name = f"{before_path.stem}_after{before_path.suffix}"
        
        # Выводим инфу, блокирует ли мебель окна
        warning = " [!] Внимание: Мебель частично перекрывает окно" if collision > 0.05 else " [OK] Мебель встает идеально"
        
        print(f" #{rank:>2} | Пол: {max(0, iou)*100:>4.1f}% | Потолок: {max(0, c_iou)*100:>4.1f}% | Глубина: {max(0, d_sim)*100:>4.1f}% | Ракурс (VP): {max(0, vp_s)*100:>4.1f}% | База: {before_path.name:25} ---> {after_name}")
        print(f"       └─ {warning} (Блокировка: {collision*100:.1f}%)")
        print("-" * 140)
    print("\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    idx_p = subparsers.add_parser("index")
    idx_p.add_argument("database_dir")

    search_p = subparsers.add_parser("search")
    search_p.add_argument("query_image")
    search_p.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "index":
        build_index(args.database_dir)
    elif args.command == "search":
        search_similar(args.query_image, top_k=args.top_k)