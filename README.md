# ✈️ AeroScan: Multimodal Edge AI Defect Detection & Structural Health Monitoring
## Tata Technologies InnoVent 2026-27 | Stage 02 Prototype
**Vertical**: Aerospace | **Category**: 3.2.3.4 (Edge AI for Intelligent Inspection & Defect Detection)  
**Institution**: Vellore Institute of Technology (VIT), Chennai | **Team**: AeroNauts  

---

### 🌟 Project Overview
**AeroScan** is an edge-native, zero-cloud multimodal inspection instrument designed to detect surface anomalies and subsurface structural micro-cracks on commercial aircraft airframes in real-time (<28ms latency).

* **Surface Optical Perception**: Ultralytics YOLOv11 ONNX INT8 (2.8 MB footprint, 24.1ms inference, 89.4% mAP@0.50).
* **Subsurface Acoustic Sensing**: 1.0 MS/s ultrasonic acoustic emission sensing via PyTorch 1D-CNN (AECNN, 37 KB footprint, 1.2ms inference).
* **Hyperbolic TDOA Multilateration**: 4-transducer spatial localization (<4.2mm error) isolating subsurface crack origin coordinates $(X, Y)$ on Al-2024-T3 alloy panels.
* **Bayesian Decision Fusion Engine**: Synchronously correlates optical bounding boxes with acoustic fracture energy into 5 deterministic MRO airworthiness action tiers.
* **Air-Gapped Audit Ledger**: Local SQLite database with cryptographic SHA-256 hash chaining meeting FAA DO-178C Level-B & DGCA Part-145 data sovereignty standards.

---

### 👥 Team AeroNauts
* **P Shreyas** *(Team Leader, B.Tech)* — Edge Embedded Systems, Hardware Deployment & Multilateration Math
* **Ashwathi Akilan** *(B.Tech)* — Acoustic Emission Signal Processing & Ultrasonic Waveform Modeling
* **Akhil S B** *(B.Tech)* — Computer Vision, Neural Compression & Edge Model Optimization (YOLOv11 INT8)
* **Lakshi A Haran** *(B.Tech)* — Multimodal Fusion Architecture & MRO Inspection HUD (DO-178C Compliance Lead)

---

### 🚀 Running the Prototype Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/lakshiaharan/aero-scan.git
   cd aero-scan
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Pipeline**:
   ```bash
   python verify_pipeline.py
   ```

4. **Launch MRO Inspection HUD**:
   ```bash
   streamlit run dashboard.py
   ```
   *(Or double-click `run_app_windows.bat` on Windows / `./run_app_linux.sh` on Linux)*

5. Open your browser at: **`http://localhost:8501`**
