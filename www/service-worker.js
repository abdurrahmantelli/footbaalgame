const CACHE_NAME = 'keeper-ai-v1';
const ASSETS_TO_CACHE = [
  './',
  'index.html',
  'manifest.json',
  'static/js/main.js',
  'static/js/physics.js',
  'static/js/keeper_ai.js',
  'static/js/scene.js',
  'models/goalkeeper_ai_9zone.tflite',
  'static/lib/tf.min.js',
  'static/lib/tf-tflite.min.js',
  'static/lib/wasm/tflite_web_api_client.js',
  'static/lib/wasm/tflite_web_api_cc.wasm',
  'static/lib/wasm/tflite_web_api_cc_simd.wasm',
  'static/lib/three.module.js'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request);
    })
  );
});