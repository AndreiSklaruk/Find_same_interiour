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
ПОИСК ПО ТОПОЛОГИИ + ГЛУБИНА + ПРОВЕРКА КОЛЛИЗИЙ МЕБЕЛИ (Staging Control).

Алгоритм:
  1. Извлекает Топологию (пустые углы и стены) из файлов before.
  2. Извлекает Depth-карту (Depth Anything V2) из файлов before.
  3. Извлекает "Отпечаток Мебели" из файлов _after.
  4. При поиске сравнивает глубину и накладывает мебель кандидата на окна запроса.
  5. Если мебель перекрывает свет/окна - штрафует и бракует вариант.
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


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
INDEX_JSON_PATH = "staging_index.json"
INDEX_FLOOR_MASKS = "staging_floor_masks.npy"
INDEX_FURNITURE_MASKS = "staging_furniture_masks.npy"
INDEX_DEPTH_MAPS = "staging_depth_maps.npy"


# ---------------------------------------------------------------------------
# 1. Загрузка модели
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

    return processor, seg_model, depth_processor, depth_model, device


# ---------------------------------------------------------------------------
# 2. КАРТА ГЛУБИНЫ
# ---------------------------------------------------------------------------
def extract_depth_map(image_path_or_pil, depth_processor, depth_model, device):
    """Извлекает нормализованную depth-карту 64×64 из изображения."""
    if isinstance(image_path_or_pil, (str, Path)):
        image = Image.open(image_path_or_pil).convert("RGB")
    else:
        image = image_path_or_pil.convert("RGB")
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)

    inputs = depth_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = depth_model(**inputs)
        depth = outputs.predicted_depth  # (1, H, W)
        # Resize до 64×64
        depth_64 = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(64, 64), mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()

    # Нормализация в [0, 1]
    d_min, d_max = depth_64.min(), depth_64.max()
    if d_max - d_min > 1e-6:
        depth_64 = (depth_64 - d_min) / (d_max - d_min)
    else:
        depth_64 = np.zeros_like(depth_64)

    del inputs, outputs, depth
    if str(device) == 'mps': torch.mps.empty_cache()
    gc.collect()

    return depth_64.astype(np.float32)


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
# 3. АНАЛИЗ ГЕОМЕТРИИ И ОТПЕЧАТКОВ МЕБЕЛИ
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
    
    # Собираем маски (64x64)
    small_floor = cv2.resize(clean_floor, (64, 64), interpolation=cv2.INTER_NEAREST)
    
    # "Запретная зона" - окна и двери, которые нельзя перекрывать мебелью
    no_go_zone = np.logical_or(window_mask, door_mask).astype(np.uint8)
    small_no_go_zone = cv2.resize(no_go_zone, (64, 64), interpolation=cv2.INTER_NEAREST)

    del inputs, outputs, logits, preds, image
    if str(device) == 'mps': torch.mps.empty_cache()
    gc.collect()

    return room_type, dominant_wall, corner_x_ratio, windows, floor_ratio, small_floor, small_no_go_zone


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
# 3. ИНДЕКСАЦИЯ БАЗЫ (BEFORE + AFTER)
# ---------------------------------------------------------------------------
def build_index(database_dir, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    database_dir = Path(database_dir)
    image_paths = sorted([p for p in database_dir.rglob("*") if p.suffix.lower() in extensions])

    if not image_paths:
        raise ValueError("Изображений не найдено.")

    processor, seg_model, depth_processor, depth_model, device = load_models()
    logical_db = {}
    floor_masks = []
    furniture_masks = []
    depth_maps = []

    for img_path in tqdm(image_paths, desc="Индексация", unit="img"):
        # Мы индексируем только пустые болванки, но подглядываем в их _after версии
        if "_after" in img_path.stem.lower():
            continue

        try:
            # 1. Анализ пустой комнаты
            r_type, dom_wall, corner_x, wins, ratio, floor_m, _ = analyze_topology(img_path, processor, seg_model, device)
            
            # 2. Извлечение карты глубины
            depth_m = extract_depth_map(img_path, depth_processor, depth_model, device)
            
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

            # Сохраняем в базу
            idx = len(floor_masks)
            logical_db[str(img_path)] = {
                "index": idx,
                "room_type": r_type,
                "dominant_wall": dom_wall,
                "corner_x": corner_x,
                "windows": wins,
                "floor_ratio": ratio
            }
            floor_masks.append(floor_m)
            furniture_masks.append(furn_m)
            depth_maps.append(depth_m)

        except Exception as e:
            print(f"[WARN] Пропущен {img_path.name}: {e}")

    with open(INDEX_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(logical_db, f, ensure_ascii=False, indent=2)
    
    np.save(INDEX_FLOOR_MASKS, np.stack(floor_masks))
    np.save(INDEX_FURNITURE_MASKS, np.stack(furniture_masks))
    np.save(INDEX_DEPTH_MAPS, np.stack(depth_maps))
    print(f"[OK] База стейджинга сохранена! ({len(logical_db)} комнат с мебелью + depth)")


# ---------------------------------------------------------------------------
# 5. ПОИСК С ПРОВЕРКОЙ КОЛЛИЗИЙ + ГЛУБИНА
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
    db_furniture_masks = np.load(INDEX_FURNITURE_MASKS)
    db_depth_maps = np.load(INDEX_DEPTH_MAPS)

    processor, seg_model, depth_processor, depth_model, device = load_models()

    print(f"\n[INFO] АНАЛИЗ ЗАПРОСА: {Path(query_image_path).name}")
    # Для запроса нам важна No-Go Zone (Окна и Двери)
    q_type, q_dom_wall, _, q_wins, q_ratio, q_floor, q_nogo = analyze_topology(query_image_path, processor, seg_model, device)
    # Извлекаем depth-карту запроса
    q_depth = extract_depth_map(query_image_path, depth_processor, depth_model, device)
    
    print(f" ├─ Тип комнаты: {'Фронтальная' if q_type == 'Frontal' else 'Угловая'}")
    print(f" ├─ Доминирует: {'Стена 1' if q_dom_wall == 'Left' else 'Стена 2' if q_dom_wall == 'Right' else 'Симметрия'}")
    print(f" ├─ Depth-карта: извлечена (64×64)")
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
    # ШАГ 2: Виртуальная примерка мебели + Глубина
    # =========================================================
    results = []
    total_nogo_pixels = np.sum(q_nogo)

    for filepath in candidates:
        data = db[filepath]
        idx = data["index"]
        
        # 1. Насколько хорошо совпал пол
        floor_iou = compute_iou(q_floor, db_floor_masks[idx])
        
        # 2. ПРОВЕРКА КОЛЛИЗИЙ (Накладываем чужую мебель на наши окна)
        candidate_furniture = db_furniture_masks[idx]
        blocked_pixels = np.logical_and(q_nogo, candidate_furniture).sum()
        
        # Какой процент окна перекрыт?
        collision_percent = (blocked_pixels / total_nogo_pixels) if total_nogo_pixels > 0 else 0.0
        
        # 3. Совпадение глубины (NCC)
        depth_sim = compute_depth_similarity(q_depth, db_depth_maps[idx])
        
        # 4. Финальный балл (Топология + Глубина - Штрафы)
        area_penalty = abs(data["floor_ratio"] - q_ratio)
        
        final_score = floor_iou + (depth_sim * 0.3) - (area_penalty * 0.5) - (collision_percent * 2.0)
        
        results.append((final_score, filepath, floor_iou, collision_percent, depth_sim))

    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:top_k]

    print(f"\n{'='*110}")
    print(f" ТОП-{top_k} ИДЕАЛЬНЫХ СОВПАДЕНИЙ С УЧЕТОМ МЕБЕЛИ И ГЛУБИНЫ:")
    print(f"{'='*110}")
    
    for rank, (score, filepath, iou, collision, d_sim) in enumerate(top_results, start=1):
        before_path = Path(filepath)
        after_name = f"{before_path.stem}_after{before_path.suffix}"
        
        # Выводим инфу, блокирует ли мебель окна
        warning = " [!] Внимание: Мебель частично перекрывает окно" if collision > 0.05 else " [OK] Мебель встает идеально"
        
        print(f" #{rank:>2} | Пол: {max(0, iou)*100:>4.1f}% | Глубина: {max(0, d_sim)*100:>4.1f}% | База: {before_path.name:25} ---> {after_name}")
        print(f"       └─ {warning} (Блокировка: {collision*100:.1f}%)")
        print("-" * 110)
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