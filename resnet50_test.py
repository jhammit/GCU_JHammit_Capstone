# JHammit ResNet50 Functional Model Test

# Import libraries
import tensorflow.keras as K
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time
import tracemalloc

# Load pre-trained ResNet50 model with ImageNet weights
model = ResNet50(weights='imagenet')

# Path to individual image of choice
img_path = '.../select.image'

# Function to process the image and track metrics
def process_image_and_track_metrics(img_path):

    # Start tracking memory
    tracemalloc.start()

    # Load the image with target size
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Expand dimensions to match the model's input shape rules
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image (normalization and scaling as required by ResNet50)
    img_array = preprocess_input(img_array)

    # Track inference start time
    start_time = time.time()

    # Perform prediction
    predictions = model.predict(img_array)

    # Track inference end time
    end_time = time.time()

    # Stop tracking memory
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Decode the predictions into human-readable labels
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Print top 3 predictions
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({score * 100:.2f}%)")

    # Calculate metrics
    inference_time = end_time - start_time
    memory_usage_increase = peak / (1024 * 1024)

    return inference_time, memory_usage_increase

# Call the function and capture metrics
inference_time, memory_usage_increase = process_image_and_track_metrics(img_path)

# Print model performance metrics
print(f"Inference Time: {inference_time:.4f} seconds")
print(f"Memory Usage Increase: {memory_usage_increase:.4f} MiB")
