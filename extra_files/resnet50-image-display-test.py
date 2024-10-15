# JHammit ResNet50 Functional Model Batch File with Imagery Display Test

# Import libraries
import os
import numpy as np
import tensorflow.keras as K
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import time  # For time tracking
from memory_profiler import memory_usage  # For memory tracking

# Load pre-trained ResNet50 model with ImageNet weights
model = ResNet50(weights='imagenet')

# Path to your folder containing images
folder_path = os.path.expanduser('/testimages')  # Adjust this path

# Create lists to store performance metrics
inference_times = []
memory_usages = []

# Function to perform image prediction
def predict_image(img_array):
    # Start inference time
    start_time = time.time()
    predictions = model.predict(img_array)
    # End inference time
    end_time = time.time()
    return predictions, end_time - start_time

# Iterate over each image in folder
for img_file in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_file)

    # Check if the file is correct format - image
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        try:
            # Load the image with target size
            img = image.load_img(img_path, target_size=(224, 224))

            # Convert the image to a numpy array
            img_array = image.img_to_array(img)

            # Expand dimensions to match the model input shape
            img_array = np.expand_dims(img_array, axis=0)

            # Preprocess the image (normalization and scaling as required by ResNet50)
            img_array = preprocess_input(img_array)

            # Memory usage before prediction
            mem_usage_before = memory_usage()[0]

            # Perform prediction and track inference time
            predictions, inference_time = predict_image(img_array)

            # Memory usage after prediction
            mem_usage_after = memory_usage()[0]

            # Store performance metrics
            inference_times.append(inference_time)
            memory_usages.append(mem_usage_after - mem_usage_before)

            # Decode the predictions into human-readable labels
            decoded_predictions = decode_predictions(predictions, top=3)[0]

            plt.figure(figsize=(8, 8))

            # Display the original image with predictions
            plt.imshow(img)
            plt.axis('off')

            # Prepare the text for the predicted labels
            prediction_text = ""
            for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                prediction_text += f"{i + 1}: {label} ({score * 100:.2f}%)\n"

            plt.title(prediction_text)
            plt.show()  # Show the image with predictions

        except Exception as e:
            print(f"Error processing {img_file}: {e}")

# Print out performance metrics for model results
print("Inference Times (seconds):", inference_times)
print("Memory Usage (MiB):", memory_usages)
print(f"Average Inference Time: {np.mean(inference_times):.4f} seconds")
print(f"Average Memory Usage Increase: {np.mean(memory_usages):.4f} MiB")
