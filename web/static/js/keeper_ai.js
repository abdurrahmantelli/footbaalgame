// keeper_ai.js - On-Device TFLite Web Goalkeeper AI (Edge AI, no server)
// Uses tfjs-tflite UMD global (window.tflite) + tfjs-core (window.tf) loaded by index.html
// NEON SIMD accelerated WebAssembly inference, fully on-device.

// Scaler params from core/scaler_params.json (preserved exactly - DO NOT CHANGE)
const SP = {
  mean: [0.018192619288069437, 4.790123455373378, 0.30312980991145116, 28.44303745886412],
  scale: [5.199367480114628, 1.7404691315073935, 25.973776995386512, 2.917719864555101]
};

export class TFLiteGoalkeeper {
  constructor(modelUrl) {
    this.modelUrl = modelUrl;
    this.means = SP.mean;
    this.scales = SP.scale;
    this.model = null;
    this.ready = false;
  }
  async load() {
    tflite.setWasmPath("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/");
    this.model = await tflite.loadTFLiteModel(this.modelUrl);
    this.ready = true;
    return this;
  }
  predict(vx, vy, spin, flickSpeed) {
    if (!this.ready) throw new Error("model not loaded");
    const norm = (v, i) => (v - this.means[i]) / this.scales[i];
    const input = tf.tensor2d([[norm(vx, 0), norm(vy, 1), norm(spin, 2), norm(flickSpeed, 3)]], [1, 4], "float32");
    const t0 = performance.now();
    const out = this.model.predict(input);
    const t1 = performance.now();
    const probs = Array.from(out.dataSync());
    const latency = t1 - t0;
    let zone = 0, best = -Infinity;
    for (let i = 0; i < probs.length; i++) { if (probs[i] > best) { best = probs[i]; zone = i; } }
    input.dispose(); out.dispose();
    return { zone, probs, latencyMs: latency };
  }
}
