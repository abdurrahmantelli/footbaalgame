# Pocket-Keeper AI 🥅

An ultra-low latency, on-device predictive goalkeeper AI optimized for Arm-based architecture. Pocket-Keeper AI reads a player's swipe (flick) trajectory within the first 15–30 ms, predicts which of the 9 goal zones the ball is heading to (including Magnus-effect curve estimation), and triggers the goalkeeper's dive in real-time.

Submission for the **Arm Create: AI Optimization Challenge** (Track 1: Optimization Output / Edge AI).

---

## 🏆 Key Optimization Metrics
- **Model Size**: `4.16 KB` (INT8 Quantized TFLite, running in-memory).
- **Inference Latency**: `~0.19 ms` (<2 ms reaction budget).
- **Validation Accuracy**: `96.89%` (on 9-zone classification).
- **Aesthetic Debugging**: Displays probability distributions, latency telemetry, and physics curves in real time.

---

## 🥅 9-Zone Goal Layout

The goalmouth is divided into 9 target zones:

```
+-------------------+-------------------+-------------------+
|  1. SOL ÜST       |  2. ORTA ÜST      |  3. SAĞ ÜST       |
|  (90 Plase)       |  (Tavan / Sert)   |  (90 Plase)       |
+-------------------+-------------------+-------------------+
|  4. SOL ORTA      |  5. TAM ORTA      |  6. SAĞ ORTA      |
|  (Yan Ağlar)      |  (Göğüs Seviyesi) |  (Yan Ağlar)      |
+-------------------+-------------------+-------------------+
|  7. SOL ALT       |  8. ORTA ALT      |  9. SAĞ ALT       |
|  (Yerden Direk)   |  (Akıllı Panenka) |  (Yerden Direk)   |
+-------------------+-------------------+-------------------+
```

---

## 📂 Project Repository Structure
- `LICENSE`: Open-source MIT License.
- `requirements.txt`: Python package list (`pygame`, `tensorflow`, `numpy`, `pandas`, `scikit-learn`).
- `core/`: Core AI & Physics Engine.
  - `generate_data.py`: Simulates 15,000 shots with gravity, drag, and Magnus effects.
  - `train_goalkeeper.py`: Train MLP model and perform INT8 Quantization.
  - `verify_scenarios.py`: Runs automated scenario checks (Panenka, Ters Köşe, etc.).
- `demo/`: Pygame interactive visual debugger.
  - `test_simulation.py`: Interactive click-and-drag flick screen.

---

## 🚀 Setup & Execution Instructions

Follow these steps to run the simulation and verify optimization stats on your device:

### 1. Install Dependencies
Make sure you have Python 3.10+ installed, then install requirements:
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
Simulate penalty kicks with Magnus aerodynamics:
```bash
python core/generate_data.py
```
This produces `core/penalty_dataset.csv`.

### 3. Train Model & Compile to TFLite
Train the predictive model and generate the INT8 quantized `.tflite` model:
```bash
python core/train_goalkeeper.py
```
This produces `core/goalkeeper_ai_9zone.tflite` and `core/scaler_params.json`.

### 4. Run Automated Test Suite
Verify model capabilities against standard test scenarios:
```bash
python core/verify_scenarios.py
```

### 5. Start Visual Debugger
Launch the Pygame interface to manually test swipe gestures, curveballs, and Panenkas:
```bash
python demo/test_simulation.py
```

---

## ⚔️ Trick Shots & Trajectory Physics
1. **TEST-01 (Ters Köşe / Feint)**: Swipe quickly towards the left bottom corner, then curl your cursor towards the right. The AI will dive to Zone 7 (Left-Bottom) using early 20ms vectors, but the Magnus effect will curl the ball into the top-right corner (Zone 3).
2. **TEST-02 (Akıllı Panenka)**: Swipe very slowly upward. The goalkeeper AI anticipates a hard-corner shot and dives left/right, leaving the center open for the soft Panenka to drop into Zone 8 (Middle-Bottom).
