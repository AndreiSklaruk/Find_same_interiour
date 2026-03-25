"""
One-time script to enrich staging_index.json with wall layout ratios.
This adds wall_left_ratio and wall_right_ratio to each entry,
computed from SegFormer segmentation (wall class = 0 in ADE20K).
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import torch
import gc

BASE_DIR = Path(__file__).parent
INDEX_JSON_PATH = BASE_DIR / "staging_index.json"

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    device = get_device()
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.to(device)
    model.eval()
    print(f"[INFO] SegFormer loaded on {device}")

    with open(INDEX_JSON_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)

    total = len(db)
    print(f"[INFO] Enriching {total} entries...")

    for i, (filepath, data) in enumerate(db.items()):
        img_path = Path(filepath)
        if not img_path.is_absolute():
            img_path = BASE_DIR / filepath

        if not img_path.exists():
            print(f"  [{i+1}/{total}] SKIP (not found): {img_path.name}")
            data["wall_left_ratio"] = 0.0
            data["wall_right_ratio"] = 0.0
            continue

        pil_image = Image.open(img_path).convert("RGB")
        pil_image.thumbnail((512, 512), Image.Resampling.LANCZOS)

        H, W = 128, 128
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = torch.nn.functional.interpolate(
                outputs.logits, size=(H, W), mode="bilinear", align_corners=False
            )
            preds = logits.argmax(dim=1).squeeze().cpu().numpy()

        wall_mask = (preds == 0).astype(np.uint8)
        mid = W // 2
        left_half_pixels = H * mid
        right_half_pixels = H * (W - mid)
        wall_left_ratio = float(np.sum(wall_mask[:, :mid])) / left_half_pixels if left_half_pixels > 0 else 0.0
        wall_right_ratio = float(np.sum(wall_mask[:, mid:])) / right_half_pixels if right_half_pixels > 0 else 0.0

        data["wall_left_ratio"] = round(wall_left_ratio, 4)
        data["wall_right_ratio"] = round(wall_right_ratio, 4)

        del inputs, outputs, logits, preds
        if str(device) == "mps":
            torch.mps.empty_cache()
        gc.collect()

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{total}] {img_path.name}: L={wall_left_ratio:.3f} R={wall_right_ratio:.3f}")

    with open(INDEX_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Enriched {total} entries. Saved to {INDEX_JSON_PATH}")

if __name__ == "__main__":
    main()
