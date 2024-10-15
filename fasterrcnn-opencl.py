# JHammit Faster R-CNN and OpenCL

# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import kagglehub
import timeit
import pyopencl as cl

# OpenCL setup
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# OpenCL kernel
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
# Compile OpenCL program for resizing
program = cl.Program(context, kernel_code).build()

# Use KaggleHub to download tensorflow Faster R-CNN model
def download_model():
    path = kagglehub.model_download("tensorflow/faster-rcnn-inception-resnet-v2/tensorFlow2/1024x1024")
    print("Path to model files:", path)
    return path

# Preprocessing imagery through OpenCL
def preprocess_image_opencl(image_path):
    # Use OpenCV to read image
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

    # Retrieve output image from the device
    output_image = np.empty(output_width * output_height * channels, dtype=np.uint8)
    cl.enqueue_copy(queue, output_image, output_image_buf).wait()

    # Reshape the flat output array to 3D image format
    output_image = output_image.reshape((output_height, output_width, channels))

    # Normalize to [0, 1] range
    output_image = output_image / 255.0

    # Add batch dimension
    output_image = np.expand_dims(output_image, axis=0)

    return output_image

# Preprocessing imagery through CPU
def preprocess_image_cpu(image_path):
    # Read image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image using OpenCV CPU method
    output_image = cv2.resize(image, (224, 224))

    # Normalize to [0, 1] range
    output_image = output_image / 255.0

    # Add batch dimension
    output_image = np.expand_dims(output_image, axis=0)

    return output_image

# Draw bounding boxes for Faster R-CNN
def draw_boxes(img, boxes, scores, classes, threshold=0.5):
    img_height, img_width, _ = img.shape

    for i in range(len(scores)):
        score = scores[i].numpy() if hasattr(scores[i], 'numpy') else scores[i]

        if np.isscalar(score) and score > threshold:
            box = boxes[i].numpy() if hasattr(boxes[i], 'numpy') else boxes[i]

            # Convert normalized box to pixel values
            ymin, xmin, ymax, xmax = (box * np.array([img_height, img_width, img_height, img_width])).astype(int)

            class_id = int(classes[i].numpy()) if hasattr(classes[i], 'numpy') else int(classes[i])

            # Draw box and label
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, f'Class: {class_id} Score: {score:.2f}',
                        (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return img

# Show image in UI with matplotlib
def display_image(image, title="Image"):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Run inference and other performance metrics
def run_inference(detector, img_tensor):
    start_time = timeit.default_timer()

    # Run inference
    output_dict = detector(img_tensor)

    # Calculate inference
    inference_time = timeit.default_timer() - start_time

    return output_dict, inference_time

def main(image_path):
    # Download and get Faster R-CNN model path
    model_path = download_model()

    # Preprocess image with OpenCL and track time
    start_opencl = timeit.default_timer()
    img_tensor_opencl = preprocess_image_opencl(image_path)
    opencl_time = timeit.default_timer() - start_opencl

    # Preprocess image with CPU and track time
    start_cpu = timeit.default_timer()
    img_tensor_cpu = preprocess_image_cpu(image_path)
    cpu_time = timeit.default_timer() - start_cpu

    # Load downloaded Faster R-CNN model
    detector = tf.saved_model.load(model_path)

    # Display preprocessed images from OpenCL and CPU
    display_image(img_tensor_opencl[0], title="Preprocessed Image (OpenCL)")
    display_image(img_tensor_cpu[0], title="Preprocessed Image (CPU)")

    # Run inference on the OpenCL preprocessed image
    output_dict_opencl, inference_time_opencl = run_inference(detector, img_tensor_opencl)

    # Run inference on the CPU preprocessed image
    output_dict_cpu, inference_time_cpu = run_inference(detector, img_tensor_cpu)

    # Print preprocessing times for OpenCL and CPU
    print(f"OpenCL preprocessing time: {opencl_time:.4f} seconds")
    print(f"CPU preprocessing time: {cpu_time:.4f} seconds")

    # Print inference times for OpenCL and CPU
    print(f"OpenCL inference time: {inference_time_opencl:.4f} seconds")
    print(f"CPU inference time: {inference_time_cpu:.4f} seconds")

    # Calculate FPS for both CPU and OpenCL
    fps_opencl = 1 / inference_time_opencl if inference_time_opencl > 0 else 0
    fps_cpu = 1 / inference_time_cpu if inference_time_cpu > 0 else 0

    # Print FPS for both CPU and OpenCL
    print(f"OpenCL FPS: {fps_opencl:.2f}")
    print(f"CPU FPS: {fps_cpu:.2f}")

    # Load OG image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Draw bounding boxes using OpenCL image output
    if len(output_dict_opencl['detection_scores']) > 0:
        img_with_boxes_opencl = draw_boxes(img.copy(),
                                           output_dict_opencl['detection_boxes'],
                                           output_dict_opencl['detection_scores'],
                                           output_dict_opencl['detection_classes'],
                                           threshold=0.5)
        plt.figure(figsize=(10, 10))
        plt.title("OpenCL Processed Image")
        plt.imshow(cv2.cvtColor(img_with_boxes_opencl, cv2.COLOR_RGB2BGR))
        plt.axis('off')
        plt.show()

    # Draw bounding boxes using CPU image output
    if len(output_dict_cpu['detection_scores']) > 0:
        img_with_boxes_cpu = draw_boxes(img.copy(),
                                        output_dict_cpu['detection_boxes'],
                                        output_dict_cpu['detection_scores'],
                                        output_dict_cpu['detection_classes'],
                                        threshold=0.5)
        plt.figure(figsize=(10, 10))
        plt.title("CPU Processed Image")
        plt.imshow(cv2.cvtColor(img_with_boxes_cpu, cv2.COLOR_RGB2BGR))
        plt.axis('off')
        plt.show()

# Run main program
if __name__ == "__main__":
    # Update image_path to desired image to test (example images available in "testimages" folder)
    image_path = '.../testimage.jpg'
    main(image_path)
