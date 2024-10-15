# JHammit Data Preprocessing

# import libraries
import cv2
import numpy as np
import pyopencl as cl

# Set up OpenCL context and queue
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

# Compile OpenCL program for resizing
program = cl.Program(context, kernel_code).build()


# Preprocessing imagery through OpenCL
def preprocess_image(image_path):
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

    # Print status
    print("Image preprocessing (with OpenCL) is successful.")
    return output_image


# Sample usage, add custom image pathing (example images available in "testimages" folder)
preprocessed_image = preprocess_image(".../insert.image.path")
