# JHammit ResNet50 and OpenCL

# Import libraries
import os
import cv2
import numpy as np
import pyopencl as cl
import time
import psutil
import tensorflow.keras as K
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import tracemalloc

# Set custom filepath to image choice (example images available in "testimages" folder)
img_path = '.../insert.image.path'

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


# Preprocessing imagery through OpenCL
def preprocess_image(image_path):
    # Read image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # OG dimensions
    input_height, input_width, channels = image.shape

    # New dimensions
    output_width, output_height = 224, 224

    # Flatten image to 1D array
    image_flattened = image.flatten().astype(np.uint8)

    # Allocate memory on device
    mf = cl.mem_flags
    input_image_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image_flattened)
    output_image_buf = cl.Buffer(context, mf.WRITE_ONLY, output_width * output_height * channels)

    # Set kernel arguments
    kernel = program.resize
    kernel.set_arg(0, input_image_buf)
    kernel.set_arg(1, output_image_buf)
    kernel.set_arg(2, np.int32(input_width))
    kernel.set_arg(3, np.int32(input_height))
    kernel.set_arg(4, np.int32(output_width))
    kernel.set_arg(5, np.int32(output_height))

    # Execute OpenCL kernel
    cl.enqueue_nd_range_kernel(queue, kernel, (output_width, output_height), None)

    # Retrieve output image
    output_image = np.empty(output_width * output_height * channels, dtype=np.uint8)
    cl.enqueue_copy(queue, output_image, output_image_buf).wait()

    # Reshape to OG dimensions
    output_image = output_image.reshape((output_height, output_width, channels))

    # Convert pixel values to [0, 255] range for ResNet50 preprocessing
    output_image = output_image.astype(np.float32)

    # Print Status
    print("Image preprocessing (with OpenCL) is successful.")
    return output_image


# Read and resize image using CPU for comparison
def cpu_resize(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized_image = cv2.resize(image, (224, 224)).astype(np.float32)
    return resized_image


# Analyze memory usage
def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB


# Load pre-trained ResNet50 model with ImageNet weights
model = ResNet50(weights='imagenet')


# Function to process the image and track metrics
def process_image_and_track_metrics(img_array):
    # Memory Tracking
    tracemalloc.start()

    # Check the shape
    print(f"Image shape for prediction: {img_array.shape}")

    # Expand dimensions to match the model expected input shape
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image
    img_array = preprocess_input(img_array)

    # Track the start time for inference
    start_time = time.time()

    # Perform prediction
    predictions = model.predict(img_array)

    # Track the end time for inference
    end_time = time.time()

    # Stop tracking memory
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Decode the predictions into human-readable labels
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Print the top 3 predictions
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({score * 100:.2f}%)")

    # Calculate metrics
    inference_time = end_time - start_time
    memory_usage_increase = peak / (1024 * 1024)  # Convert bytes to MiB

    return inference_time, memory_usage_increase


# Main logic to compare OpenCL and CPU-based preprocessing
if __name__ == '__main__':
    if os.path.exists(img_path):
        # OpenCL Processing
        print("\nStarting OpenCL-based preprocessing...")
        mem_usage_before_opencl = memory_usage()
        opencl_image = preprocess_image(img_path)
        mem_usage_after_opencl = memory_usage()
        print(f"OpenCL Memory Usage: Before: {mem_usage_before_opencl:.2f} MB, After: {mem_usage_after_opencl:.2f} MB")

        if opencl_image is not None:
            # Process OpenCL image
            print("\nProcessing OpenCL Image:")
            opencl_inference_time, opencl_memory_increase = process_image_and_track_metrics(opencl_image)
            print(f"OpenCL Inference Time: {opencl_inference_time:.4f} seconds")
            print(f"OpenCL Memory Usage Increase: {opencl_memory_increase:.4f} MiB")
        else:
            print("OpenCL image preprocessing failed. Exiting.")

        # CPU Processing
        print("\nStarting CPU-based preprocessing...")
        mem_usage_before_cpu = memory_usage()
        cpu_image = cpu_resize(img_path)
        mem_usage_after_cpu = memory_usage()
        print(f"CPU Memory Usage: Before: {mem_usage_before_cpu:.2f} MB, After: {mem_usage_after_cpu:.2f} MB")

        if cpu_image is not None:
            # Process CPU image
            print("\nProcessing CPU Image:")
            cpu_inference_time, cpu_memory_increase = process_image_and_track_metrics(cpu_image)
            print(f"CPU Inference Time: {cpu_inference_time:.4f} seconds")
            print(f"CPU Memory Usage Increase: {cpu_memory_increase:.4f} MiB")
        else:
            print("CPU image preprocessing failed. Exiting.")