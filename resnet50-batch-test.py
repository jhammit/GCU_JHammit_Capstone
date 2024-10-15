# JHammit ResNet50 Functional Model Batch File Test

# Import libraries
import os
import numpy as np
import tensorflow.keras as K
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load pre-trained ResNet50 model with ImageNet weights
model = ResNet50(weights='imagenet')

# Absolute path to folder containing image files, update path as needed
folder_path = os.path.expanduser('/testimages')

# Iterate over all files in the folder
for img_file in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_file)

    # Check if the file is an image file
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        try:
            # Load the image with target size
            img = image.load_img(img_path, target_size=(224, 224))

            # Convert the image to a numpy array
            img_array = image.img_to_array(img)

            # Expand dimensions to match the model's input shape
            img_array = np.expand_dims(img_array, axis=0)

            # Preprocess the image (normalization and scaling as required by ResNet50)
            img_array = preprocess_input(img_array)

            # Perform prediction
            predictions = model.predict(img_array)

            # Decode the predictions into human-readable labels
            decoded_predictions = decode_predictions(predictions, top=3)[0]

            # Print the filename and the top 3 predictions
            print(f"Predictions for {img_file}:")
            for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                print(f"  {i + 1}: {label} ({score * 100:.2f}%)")
            print()
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
