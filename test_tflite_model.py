import tensorflow as tf
import numpy as np
from PIL import Image
import time

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def load_tflite_model(model_path='recyclable_classifier.tflite'):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image_path, img_size=224):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

def predict_image(interpreter, image_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    img = preprocess_image(image_path)
    
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    inference_time = (time.time() - start_time) * 1000
    
    predicted_class = np.argmax(output[0])
    confidence = output[0][predicted_class]
    
    print(f"Image: {image_path}")
    print(f"Predicted: {CLASS_NAMES[predicted_class]}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Inference time: {inference_time:.2f}ms\n")
    
    return CLASS_NAMES[predicted_class], confidence, inference_time

def benchmark_model(interpreter, test_images):
    inference_times = []
    
    for img_path in test_images:
        _, _, inf_time = predict_image(interpreter, img_path)
        inference_times.append(inf_time)
    
    avg_time = np.mean(inference_times)
    print(f"Average inference time: {avg_time:.2f}ms")
    print(f"Min: {min(inference_times):.2f}ms, Max: {max(inference_times):.2f}ms")

if __name__ == '__main__':
    interpreter = load_tflite_model()
    test_images = ['test1.jpg', 'test2.jpg', 'test3.jpg']
    benchmark_model(interpreter, test_images)