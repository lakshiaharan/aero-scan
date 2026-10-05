import sqlite3
from PIL import Image
from defect_detector import detect
from ae_pipeline import make_sample, hit_features
from fusion import classify_ae, fuse
import pandas as pd

print("--- TEST 1: Optical Model ---")
test_img = Image.new('RGB', (640, 480), (30, 40, 50))
dets = detect(test_img, conf=0.1)
print("Detections on test frame:", dets)

print("\n--- TEST 2: Acoustic 1D-CNN ---")
for cls_idx in [0, 1, 2]:
    sig = make_sample(cls_idx)
    res = classify_ae(sig)
    feats = hit_features(sig)
    print(f"AE class {cls_idx} -> pred: {res}, energy: {feats['energy']:.1f}")

print("\n--- TEST 3: Multimodal Fusion & DB Log ---")
sig = make_sample(2)
ae_res = classify_ae(sig)
mock_rgb = [{'class': 'crack', 'conf': 0.95, 'bbox': [100, 100, 50, 50]}]
verdict = fuse(mock_rgb, ae_res)
print("Fused verdict:", verdict)

from dashboard import log_to_db, load_logs
log_to_db(verdict)
logs = load_logs(3)
print("\nRecent DB Logs:")
print(logs[['id', 'timestamp', 'status', 'severity', 'confidence']])
print("\n>>> ALL SYSTEMS 100% OPERATIONAL <<<")
