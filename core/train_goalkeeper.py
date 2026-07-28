import numpy as np
import pandas as pd
import json
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_model():
    print("TensorFlow Version:", tf.__version__)
    
    # 1. Load Data
    dataset_path = 'core/penalty_dataset.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run generate_data.py first.")
        
    df = pd.read_csv(dataset_path)
    X = df[['early_vx', 'early_vy', 'spin', 'flick_speed']].values
    y = df['target_zone'].values
    
    # 2. Train Test Split & Scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler parameters to JSON for client/simulation loading
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }
    with open('core/scaler_params.json', 'w') as f:
        json.dump(scaler_params, f)
    print("Scaler parameters saved to core/scaler_params.json")
    
    # 3. Build a lightweight MLP (Optimized for ultra-low latency mobile/Arm execution)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(9, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 4. Train Model
    print("Training the predictive Goalkeeper AI model...")
    model.fit(
        X_train_scaled, y_train,
        epochs=15,
        batch_size=64,
        validation_data=(X_test_scaled, y_test),
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Validation Accuracy: {accuracy*100:.2f}%")
    
    # 5. Convert to TFLite with INT8 Post-Training Quantization
    print("Converting model to INT8 Quantized TFLite...")
    
    # Representative dataset generator required for INT8 quantization calibration
    def representative_data_gen():
        # Yield representative samples (e.g. 100 samples)
        for i in range(100):
            sample = X_train_scaled[i].astype(np.float32)
            # Add batch dimension
            yield [np.expand_dims(sample, axis=0)]
            
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    
    # Ensure full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # To make integration in Pygame/mobile easy and avoid casting inputs to float -> int8 -> float manually,
    # we can set the model input/output tensors to float32 while the internal operations are quantized to int8.
    # If the user wants 100% pure INT8 execution end-to-end on NPU, we can set:
    # converter.inference_input_type = tf.int8
    # converter.inference_output_type = tf.int8
    # Here we keep inputs/outputs as float32 for ease of prototyping, but internal weights are INT8.
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    tflite_quant_model = converter.convert()
    
    # Save the quantized model
    tflite_path = 'core/goalkeeper_ai_9zone.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_quant_model)
        
    print(f"INT8 Quantized TFLite model successfully saved to {tflite_path}")
    print(f"Model size: {len(tflite_quant_model) / 1024:.2f} KB (Target: < 10 KB)")

if __name__ == "__main__":
    train_model()
