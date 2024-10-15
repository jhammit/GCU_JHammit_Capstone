import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import timeit
import os
import pyopencl as cl
import psutil  # For memory usage analysis
import zipfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# OpenCL implementation
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# OpenCL kernel for resizing
kernel_code = """
__kernel void resize(__global const uchar *input_image,
                     __global uchar *output_image,
                     const int input_width,
                     const int input_height,
                     const int output_width,
                     const int output_height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= output_width || y >= output_height) return;

    int src_x = (int)((float)x / output_width * input_width);
    int src_y = (int)((float)y / output_height * input_height);

    int src_idx = (src_y * input_width + src_x) * 3; // Assuming 3 channels (RGB)
    int dst_idx = (y * output_width + x) * 3;

    output_image[dst_idx] = input_image[src_idx];
    output_image[dst_idx + 1] = input_image[src_idx + 1];
    output_image[dst_idx + 2] = input_image[src_idx + 2];
}
"""

# Compile the OpenCL program
program = cl.Program(context, kernel_code).build()


def preprocess_image(image, use_opencl=True):
    """Preprocess the image using OpenCL or CPU."""
    if use_opencl:
        # OpenCL preprocessing
        input_height, input_width, channels = image.shape
        output_width, output_height = 224, 224
        image_flattened = image.flatten().astype(np.uint8)

        # Allocate memory on device
        mf = cl.mem_flags
        input_image_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image_flattened)
        output_image_buf = cl.Buffer(context, mf.WRITE_ONLY, output_width * output_height * channels)

        # Set kernel arguments and execute
        kernel = program.resize
        kernel.set_arg(0, input_image_buf)
        kernel.set_arg(1, output_image_buf)
        kernel.set_arg(2, np.int32(input_width))
        kernel.set_arg(3, np.int32(input_height))
        kernel.set_arg(4, np.int32(output_width))
        kernel.set_arg(5, np.int32(output_height))
        cl.enqueue_nd_range_kernel(queue, kernel, (output_width, output_height), None)

        # Retrieve output image
        output_image = np.empty(output_width * output_height * channels, dtype=np.uint8)
        cl.enqueue_copy(queue, output_image, output_image_buf).wait()
        output_image = output_image.reshape((output_height, output_width, channels)) / 255.0
        return output_image
    else:
        # CPU preprocessing
        return cv2.resize(image, (224, 224)) / 255.0


def download_model():
    """Download the latest version of the Faster R-CNN model."""
    model_dir = "../models"  # Directory to save the model
    os.makedirs(model_dir, exist_ok=True)

    try:
        logging.info("Downloading model...")
        os.system("kaggle datasets download -d tensorflow/faster-rcnn-inception-resnet-v2 -p models")
        logging.info("Model downloaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error downloading model: {str(e)}")

    # Unzip the downloaded model
    try:
        with zipfile.ZipFile(os.path.join(model_dir, "faster-rcnn-inception-resnet-v2.zip"), 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        logging.info("Model extracted successfully.")
    except Exception as e:
        raise RuntimeError(f"Error extracting model: {str(e)}")

    # Load the model
    model_path = os.path.join(model_dir,
                              "faster_rcnn_inception_resnet_v2.pb")  # Adjust based on the extracted model file
    try:
        detector = tf.saved_model.load(model_path)
        logging.info("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

    return detector


def load_image(image_file):
    """Convert uploaded image to RGB format."""
    image = Image.open(image_file)
    img_rgb = np.array(image.convert("RGB"))  # Convert to RGB
    return img_rgb


def run_inference(detector, img_tensor):
    """Run inference and measure performance metrics."""
    start_time = timeit.default_timer()
    output_dict = detector(img_tensor)
    inference_time = timeit.default_timer() - start_time
    return output_dict, inference_time


def draw_boxes(img, boxes, scores, classes, threshold=0.5):
    """Draw bounding boxes on the image."""
    img_height, img_width, _ = img.shape
    for i in range(len(scores)):
        score = scores[i].numpy() if hasattr(scores[i], 'numpy') else scores[i]
        if np.isscalar(score) and score > threshold:
            box = boxes[i].numpy() if hasattr(boxes[i], 'numpy') else boxes[i]
            ymin, xmin, ymax, xmax = (box * np.array([img_height, img_width, img_height, img_width])).astype(int)
            class_id = int(classes[i].numpy()) if hasattr(classes[i], 'numpy') else int(classes[i])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, f'Class: {class_id} Score: {score:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 2)
    return img


def main():
    st.title("Object Detection App with Image Processing")

    # Upload JPEG Image
    uploaded_file = st.file_uploader("Choose a JPEG image", type=["jpeg", "jpg"])
    if uploaded_file is not None:
        # Load and display the image
        img_rgb = load_image(uploaded_file)
        st.image(img_rgb, caption='Uploaded Image', use_column_width=True)

        # Preprocess image using OpenCL
        start_time_opencl = timeit.default_timer()
        preprocessed_image_opencl = preprocess_image(img_rgb, use_opencl=True)
        opencl_time = timeit.default_timer() - start_time_opencl

        # Preprocess image using CPU
        start_time_cpu = timeit.default_timer()
        preprocessed_image_cpu = preprocess_image(img_rgb, use_opencl=False)
        cpu_time = timeit.default_timer() - start_time_cpu

        # Load TensorFlow model
        try:
            detector = download_model()
        except RuntimeError as e:
            st.error(f"Error loading model: {str(e)}")
            return

        # Run inference with OpenCL preprocessed image
        img_tensor_opencl = np.expand_dims(preprocessed_image_opencl, axis=0)  # Add batch dimension
        output_dict_opencl, inference_time_opencl = run_inference(detector, img_tensor_opencl)

        # Run inference with CPU preprocessed image
        img_tensor_cpu = np.expand_dims(preprocessed_image_cpu, axis=0)  # Add batch dimension
        output_dict_cpu, inference_time_cpu = run_inference(detector, img_tensor_cpu)

        # Extract boxes, scores, and classes from OpenCL output
        boxes_opencl = output_dict_opencl['detection_boxes']
        scores_opencl = output_dict_opencl['detection_scores']
        classes_opencl = output_dict_opencl['detection_classes']

        # Extract boxes, scores, and classes from CPU output
        boxes_cpu = output_dict_cpu['detection_boxes']
        scores_cpu = output_dict_cpu['detection_scores']
        classes_cpu = output_dict_cpu['detection_classes']

        # Draw bounding boxes and display image with detections for OpenCL
        img_with_boxes_opencl = draw_boxes(preprocessed_image_opencl, boxes_opencl, scores_opencl, classes_opencl)
        st.image(img_with_boxes_opencl, caption='Image with Detections (OpenCL)', use_column_width=True)

        # Draw bounding boxes and display image with detections for CPU
        img_with_boxes_cpu = draw_boxes(preprocessed_image_cpu, boxes_cpu, scores_cpu, classes_cpu)
        st.image(img_with_boxes_cpu, caption='Image with Detections (CPU)', use_column_width=True)

        # Display inference performance metrics
        st.write(f"Inference Time (OpenCL): {inference_time_opencl:.4f} seconds")
        st.write(f"Inference Time (CPU): {inference_time_cpu:.4f} seconds")
        st.write(f"Preprocessing Time (OpenCL): {opencl_time:.4f} seconds")
        st.write(f"Preprocessing Time (CPU): {cpu_time:.4f} seconds")


if __name__ == "__main__":
    main()
