# JHammit Faster R-CNN Functional Model Test

# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import kagglehub
import timeit

# Download model from Kagglehub
def download_model():
    path = kagglehub.model_download("tensorflow/faster-rcnn-inception-resnet-v2/tensorFlow2/1024x1024")
    print("Path to model files:", path)
    return path

# Load image
def load_image(image_path):
    # Will need to update image path
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb

# Create bounding boxes for inputted image
def draw_boxes(img, boxes, scores, classes, threshold=0.5):
    img_height, img_width, _ = img.shape

    for i in range(len(scores)):
        score = scores[i].numpy() if hasattr(scores[i], 'numpy') else scores[i]

        if np.isscalar(score) and score > threshold:
            box = boxes[i].numpy() if hasattr(boxes[i], 'numpy') else boxes[i]

            # Convert normalized box to pixel values
            ymin, xmin, ymax, xmax = (box * np.array([img_height, img_width, img_height, img_width])).astype(int)

            class_id = int(classes[i].numpy()) if hasattr(classes[i], 'numpy') else int(classes[i])

            # Draw bounding box and label
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, f'Class: {class_id} Score: {score:.2f}',
                        (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return img

# Regular processing for image based on model needs
def preprocess_image(img_rgb):
    img_array = np.expand_dims(img_rgb, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Run inference and performance metrics analysis
def run_inference(detector, img_tensor):
    start_time = timeit.default_timer()

    # Execute the detector to run inference
    output_dict = detector(img_tensor)

    # Calculate inference time
    inference_time = timeit.default_timer() - start_time

    return output_dict, inference_time

# Run Main
def main(image_path):
    # Download and get the model path from 'models'
    model_path = download_model()

    # Load and preprocess image
    img, img_rgb = load_image(image_path)
    img_tensor = preprocess_image(img_rgb)

    # Load the downloaded Faster R-CNN model - Kagglehub
    detector = tf.saved_model.load(model_path)

    # Run inference
    output_dict, inference_time = run_inference(detector, img_tensor)

    # Extract outputs
    boxes = output_dict['detection_boxes']
    scores = output_dict['detection_scores']
    class_ids = output_dict['detection_classes']

    # Ensure to check dimensions before using
    if len(scores) > 0 and len(boxes) > 0:
        # Print shapes and values of the outputs for debugging
        print("Boxes shape:", boxes.shape)
        print("Scores shape:", scores.shape)
        print("Class IDs shape:", class_ids.shape)

        # Draw bounding boxes
        img_with_boxes = draw_boxes(img, boxes, scores, class_ids, threshold=0.5)

        # Display image
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        print("No detections found.")

    # Print performance metrics
    print(f"Inference time: {inference_time:.4f} seconds")
    fps = 1 / inference_time if inference_time > 0 else 0
    print(f"Frames per second (FPS): {fps:.2f}")


if __name__ == "__main__":
    # Update image path with correct image
    image_path = ".../image.of.choice"
    main(image_path)
