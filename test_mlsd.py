"""Тест: Depth Anything V2 → MLSD = чистая геометрия комнаты!"""
import cv2, numpy as np, torch, os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from mlsd.model import MobileV2_MLSD_Tiny
from mlsd.utils import pred_lines
from image_retrieval import find_vanishing_point

# 1. Depth Anything V2
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
depth_proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
depth_model.to(device).eval()

pil_img = Image.open("test_room.jpg").convert("RGB")
inputs = depth_proc(images=pil_img, return_tensors="pt").to(device)
with torch.no_grad():
    depth_out = depth_model(**inputs).predicted_depth
    depth = depth_out.squeeze().cpu().numpy()

# Нормализуем depth в 0-255 для визуализации и MLSD
depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255
depth_u8 = depth_norm.astype(np.uint8)

# Конвертируем в RGB (MLSD ожидает RGB)
depth_rgb = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2RGB)

# Сохраним depth-карту
cv2.imwrite("test_room_depth.jpg", depth_u8)
print(f"Depth map: {depth_u8.shape}")

# 2. MLSD на depth-карте
mlsd = MobileV2_MLSD_Tiny()
mlsd.load_state_dict(torch.load("mlsd/weights/mlsd_tiny_512_fp32.pth", map_location="cpu"), strict=True)
mlsd.eval()

h, w = depth_rgb.shape[:2]
lines = pred_lines(depth_rgb, mlsd, torch.device("cpu"), [512, 512], score_thr=0.10, dist_thr=20.0)
print(f"MLSD на depth: {len(lines)} линий")

if len(lines) > 0:
    vx, vy = find_vanishing_point(lines, w, h)
    print(f"VP = ({vx:.3f}, {vy:.3f})")

    # Визуализация на чёрном фоне (как в сервисе)
    vis_black = np.zeros_like(depth_rgb)
    for l in lines:
        cv2.line(vis_black, (int(l[0]),int(l[1])), (int(l[2]),int(l[3])), (255,255,255), 2, cv2.LINE_AA)
    cv2.imwrite("test_room_depth_mlsd_black.jpg", vis_black)

    # Визуализация на оригинальной фотографии
    orig = cv2.imread("test_room.jpg")
    oh, ow = orig.shape[:2]
    for l in lines:
        # Масштабируем из depth-пространства в оригинал
        x1 = int(l[0] / w * ow)
        y1 = int(l[1] / h * oh)
        x2 = int(l[2] / w * ow)
        y2 = int(l[3] / h * oh)
        cv2.line(orig, (x1,y1), (x2,y2), (0,255,0), 2, cv2.LINE_AA)
    px, py = int(vx*ow), int(vy*oh)
    if 0<=px<ow and 0<=py<oh:
        cv2.circle(orig, (px,py), 15, (0,0,255), 3)
        cv2.circle(orig, (px,py), 5, (0,0,255), -1)
        cv2.line(orig, (px-25,py), (px+25,py), (0,0,255), 2)
        cv2.line(orig, (px,py-25), (px,py+25), (0,0,255), 2)
    cv2.putText(orig, f"Depth+MLSD: {len(lines)} lines", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(orig, f"VP=({vx:.3f},{vy:.3f})", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.imwrite("test_room_depth_mlsd_overlay.jpg", orig)
    
    print("[OK] Saved: test_room_depth_mlsd_black.jpg + test_room_depth_mlsd_overlay.jpg")
else:
    print("Линий не найдено")
