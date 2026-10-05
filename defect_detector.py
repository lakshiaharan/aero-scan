# defect_detector.py — Optical Perception Inference Core (YOLOv11 INT8)
# Team AeroNauts // Tata Technologies InnoVent Stage 02
import os
from pathlib import Path

# Resolve local weights relative to this script
BASE_DIR = Path(__file__).resolve().parent
LOCAL_ONNX = BASE_DIR / "best.onnx"
LOCAL_PT = BASE_DIR / "best.pt"

_model = None

def get_model():
    """Lazily loads and caches the YOLOv11 model with robust fallback."""
    global _model
    if _model is not None:
        return _model
    try:
        from ultralytics import YOLO
        if LOCAL_ONNX.exists():
            _model = YOLO(str(LOCAL_ONNX), task='detect')
        elif LOCAL_PT.exists():
            _model = YOLO(str(LOCAL_PT))
        else:
            _model = YOLO("best.onnx", task='detect')
        return _model
    except Exception as e:
        print(f"Warning: YOLOv11 model load fallback: {e}")
        return None

def detect(frame, conf=0.25):
    """
    frame: image path, PIL Image, OR a numpy array (e.g. cv2.imread / a camera frame).
    returns: list of {"class": str, "conf": float, "bbox": [x, y, w, h]}
             bbox = top-left x, y + width, height, in pixels.
    """
    m = get_model()
    if m is None:
        return []
    try:
        results = m.predict(frame, conf=conf, verbose=False)
        out = []
        for r in results:
            for box in r.boxes:
                cx, cy, w, h = box.xywh[0].tolist()        # center-based xywh
                out.append({
                    "class": m.names[int(box.cls[0])],
                    "conf":  round(float(box.conf[0]), 3),
                    "bbox":  [round(cx - w/2, 1), round(cy - h/2, 1), round(w, 1), round(h, 1)],
                })
        return out
    except Exception as e:
        print(f"Inference error in detect(): {e}")
        return []

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for d in detect(sys.argv[1]):
            print(d)
    else:
        print(f"defect_detector loaded successfully. Model status: {get_model()}")
