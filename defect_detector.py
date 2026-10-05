# defect_detector.py — Person 1's shared inference interface (Step 7)
# Person 3's dashboard + the fusion layer import detect() from here.
from pathlib import Path
from ultralytics import YOLO

# Resolve local weights relative to this script
BASE_DIR = Path(__file__).resolve().parent
LOCAL_PT = BASE_DIR / "best.pt"
LOCAL_ONNX = BASE_DIR / "best.onnx"

if LOCAL_PT.exists():
    MODEL_PATH = str(LOCAL_PT)
elif LOCAL_ONNX.exists():
    MODEL_PATH = str(LOCAL_ONNX)
else:
    MODEL_PATH = "best.pt"

_model = YOLO(MODEL_PATH)   # loaded once at import

def detect(frame, conf=0.25):
    """
    frame: image path, PIL Image, OR a numpy array (e.g. cv2.imread / a camera frame).
    returns: list of {"class": str, "conf": float, "bbox": [x, y, w, h]}
             bbox = top-left x, y + width, height, in pixels.
    """
    results = _model.predict(frame, conf=conf, verbose=False)
    out = []
    for r in results:
        for box in r.boxes:
            cx, cy, w, h = box.xywh[0].tolist()        # center-based xywh
            out.append({
                "class": _model.names[int(box.cls[0])],
                "conf":  round(float(box.conf[0]), 3),
                "bbox":  [round(cx - w/2, 1), round(cy - h/2, 1), round(w, 1), round(h, 1)],
            })
    return out

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for d in detect(sys.argv[1]):
            print(d)
    else:
        print(f"defect_detector loaded successfully with model: {MODEL_PATH}")

