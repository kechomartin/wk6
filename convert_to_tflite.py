import tensorflow as tf
import os

def convert_to_tflite(model_path='recyclable_classifier.h5', quantize=True):
    model = tf.keras.models.load_model(model_path)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    output_path = 'recyclable_classifier.tflite'
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    tflite_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"TFLite model size: {tflite_size:.2f} MB")
    print(f"Size reduction: {((original_size - tflite_size) / original_size) * 100:.2f}%")
    
    return output_path

if __name__ == '__main__':
    convert_to_tflite()