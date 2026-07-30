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

### 0. Web Build (Commit: ba2115f)
This build is optimized for local static server execution using the TFLite Web runtime.

```bash
# From the repo root (Commit: ba2115f)
# Ensure you are on the commit that enables WebGL/WASM support
python -m http.server 8000
```
Then visit `http://localhost:8000/www/index.html` in your browser. 
**Note**: Ensure your browser supports WASM SIMD for optimal performance.

---

# 📱 Mobil Cihaz Kurulum & APK Derleme Rehberi

## 1. Web Tabanlı Erişim (WiFi)
Bilgisayarınızdaki sunucuyu telefonunuzdan görüntülemek için:
- **Sunucu**: `python -m http.server 8000`
- **Adres**: `http://<BILGISAYAR_IP>:8000/web/templates/index.html`

## 2. Android APK Derleme
Projeyi yerel bir Android uygulaması olarak çalıştırmak için aşağıdaki adımları izleyin:

### Hazırlık
```bash
# 1. Proje bağımlılıklarını kurun
npm install

# 2. Web projesini derleyin
npm run build

# 3. Capacitor ile Android projesini senkronize edin
npx cap sync android
```

### APK Oluşturma
```bash
# 4. Android klasörüne gidin
cd android

# 5. Gradle ile APK build işlemini başlatın
gradlew assembleDebug
```
- Çıktı dosyasını `android/app/build/outputs/apk/debug/app-debug.apk` konumunda bulabilirsiniz.

## Telefonunuzda Çalıştırma Adımları

### 1. **Sunucu Başlatma**
```bash
python -m http.server 8000
```
- **Amaç**: Web sunucusu, telefonunuzun erişebileceği statik dosyaları sunar

### 2. **Ağ Bağlantısı**
- Telefon ve bilgisayar **aynı WiFi ağında** olmalıdır
- Bilgisayarınızın yerel IP adresini bulun:

**Windows:**
```bash
ipconfig
```
IPv4 adresini not edin (örn: `192.168.1.100`)

**Mac/Linux:**
```bash
ifconfig
```

### 3. **Telefonunuzdan Erişim**
Tarayıcınızda şu adrese gidin:
```
http://<BILGISAYAR_IP>:8000/web/templates/index.html
```
Örnek: `http://192.168.1.100:8000/web/templates/index.html`

### 4. **Kontrol**
- Dokunmatik ekranda herhangi bir yere tıklayın ve sürükleyin (flick hareketi)
- Kaleci yapay zeka tahminine göre hareket edecek
- Debug paneli (mobilde alttan açılır) gerçek zamanlı telemetri gösterir

---



## 🎯 Ek Bilgiler

- Model boyutu: **4.16 KB** (INT8 Quantized TFLite)
- Tahmin süresi: **~0.19 ms**
- Doğruluk: **96.89%**
- 9 bölge sınıflandırması yapar
- Arm cihazları için optimize edilmiş

---

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
