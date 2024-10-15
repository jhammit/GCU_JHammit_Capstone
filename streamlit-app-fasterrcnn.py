# JHammit Streamlit App with Faster R-CNN

#import libraries
import streamlit as st
from streamlit_navigation_bar import st_navbar
from PIL import Image
import numpy as np
import cv2
import pyopencl as cl
import tensorflow as tf
import streamlit_authenticator as stauth
import kagglehub
import timeit
import time
import matplotlib.pyplot as plt
import tracemalloc
import yaml
from yaml.loader import SafeLoader


# Load pre-trained Faster R-CNN model
def load_model(model_path):
    return tf.saved_model.load(model_path)

# Download the Faster R-CNN model from Kaggle
def download_model():
    path = kagglehub.model_download("tensorflow/faster-rcnn-inception-resnet-v2/tensorFlow2/1024x1024")
    print("Path to model files:", path)
    return path


# Preprocess image with OpenCL
def preprocess_image_opencl(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_height, input_width, channels = image.shape
    output_width, output_height = 224, 224

    image_flattened = image.flatten().astype(np.uint8)
    context = cl.Context([cl.get_platforms()[0].get_devices()[0]])
    queue = cl.CommandQueue(context)

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

        int src_idx = (src_y * input_width + src_x) * 3;
        int dst_idx = (y * output_width + x) * 3;

        output_image[dst_idx] = input_image[src_idx];
        output_image[dst_idx + 1] = input_image[src_idx + 1];
        output_image[dst_idx + 2] = input_image[src_idx + 2];
    }
    """

    program = cl.Program(context, kernel_code).build()
    input_image_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=image_flattened)
    output_image_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, output_width * output_height * channels)

    kernel = program.resize
    kernel.set_arg(0, input_image_buf)
    kernel.set_arg(1, output_image_buf)
    kernel.set_arg(2, np.int32(input_width))
    kernel.set_arg(3, np.int32(input_height))
    kernel.set_arg(4, np.int32(output_width))
    kernel.set_arg(5, np.int32(output_height))

    cl.enqueue_nd_range_kernel(queue, kernel, (output_width, output_height), None)
    output_image = np.empty(output_width * output_height * channels, dtype=np.uint8)
    cl.enqueue_copy(queue, output_image, output_image_buf).wait()

    return output_image.reshape((output_height, output_width, channels))


# Run inference for Faster R-CNN model
def run_inference(detector, img_tensor):
    output_dict = detector(img_tensor)
    return output_dict


# Create bounding boxes for imagery
def draw_boxes(img, boxes, scores, classes, threshold=0.5):
    img_height, img_width, _ = img.shape
    for i in range(len(scores)):
        if scores[i] > threshold:
            box = boxes[i]
            ymin, xmin, ymax, xmax = (box * np.array([img_height, img_width, img_height, img_width])).astype(int)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    return img


# Metrics tracking function
def process_image_and_track_metrics(image, detector):
    # Convert image to tensor
    img_tensor = tf.convert_to_tensor(image)
    img_tensor = img_tensor[tf.newaxis, ...]

    # Track memory usage
    tracemalloc.start()

    # Run inference
    start_time = time.time()
    output_dict = run_inference(detector, img_tensor)
    inference_time = time.time() - start_time

    # Track memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Get predictions
    boxes = output_dict['detection_boxes'][0].numpy()
    scores = output_dict['detection_scores'][0].numpy()
    classes = output_dict['detection_classes'][0].numpy()

    predictions = [(int(classes[i]), boxes[i], scores[i]) for i in range(len(scores)) if scores[i] > 0.5]

    return predictions, inference_time, (peak / 10 ** 6)

# Run main w/ streamlit
def main():
    # Initialize session state for streamlit and authorization
    if 'authentication_status' not in st.session_state:
        st.session_state['authentication_status'] = None
    if 'name' not in st.session_state:
        st.session_state['name'] = ""

    # Load YAML configuration: update yaml file location below:
    with open('...yaml.location.update', 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=SafeLoader)

    # Create authenticator
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    # Handle user login
    result = authenticator.login(location='main')

    # Load Faster R-CNN  model
    model_path = download_model()
    detector = load_model(model_path)

    # If user is logged in
    if st.session_state['authentication_status']:
        pages = ["Home", "User Guide", "Help"]
        authenticator.logout()
        st.write("")

        # Define the styles with the updated navbar background color
        styles = {
            "nav": {
                "background-color": "rgb(109, 119, 93)",
            },
            "div": {
                "max-width": "32rem",
            },
            "span": {
                "border-radius": "0.5rem",
                "color": "rgb(49, 51, 63)",
                "margin": "0 0.125rem",
                "padding": "0.4375rem 0.625rem",
            },
            "active": {
                "background-color": "rgba(255, 255, 255, 0.25)",
            },
            "hover": {
                "background-color": "rgba(255, 255, 255, 0.35)",
            },
        }

        # Create the navbar with the defined styles and pages
        page = st_navbar(pages, styles=styles, key="main_navbar")

        # Display the current page
        st.title("Joffrey Hammit's Capstone Application: OpenCL Preprocessing for Faster R-CNN")

        # Conditionally display for home page
        if page == "Home":
            st.write("Thank you for visiting my application!")
            st.write("Upload an image here to test the OpenCL Preprocessing for Faster R-CNN:")
            st.write("JPEG File Upload")

            # File upload, specifically jpegs
            uploaded_file = st.file_uploader("Choose a JPEG file", type=["jpeg", "jpg"])

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded JPEG Image', use_column_width=True)

                # Convert the uploaded image to a format suitable for processing
                image_np = np.array(image)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCL

                # OpenCL Processing
                st.subheader("Processing with OpenCL...")
                opencl_image = preprocess_image_opencl(image_np)
                opencl_predictions, opencl_inference_time, opencl_memory_increase = process_image_and_track_metrics(
                    opencl_image, detector)

                # PRint OpenCL Results
                st.write("**OpenCL Predictions:**")
                for i, (imagenet_id, box, score) in enumerate(opencl_predictions):
                    st.write(f"{i + 1}: Class {imagenet_id} ({score * 100:.2f}%)")
                st.write(f"OpenCL Inference Time: {opencl_inference_time:.4f} seconds")
                st.write(f"OpenCL Memory Usage Increase: {opencl_memory_increase:.4f} MiB")

                # CPU Processing
                st.subheader("Processing with CPU...")
                cpu_image = preprocess_image_cpu(image_np)
                cpu_predictions, cpu_inference_time, cpu_memory_increase = process_image_and_track_metrics(cpu_image,
                                                                                                           detector)
                # Print CPU results
                st.write("**CPU Predictions:**")
                for i, (imagenet_id, box, score) in enumerate(cpu_predictions):
                    st.write(f"{i + 1}: Class {imagenet_id} ({score * 100:.2f}%)")
                st.write(f"CPU Inference Time: {cpu_inference_time:.4f} seconds")
                st.write(f"CPU Memory Usage Increase: {cpu_memory_increase:.4f} MiB")

                # Draw bounding boxes on OpenCL and CPU images
                opencl_output_image = draw_boxes(opencl_image, *zip(*opencl_predictions))
                cpu_output_image = draw_boxes(cpu_image, *zip(*cpu_predictions))

                st.image(opencl_output_image, caption='OpenCL Output Image', use_column_width=True)
                st.image(cpu_output_image, caption='CPU Output Image', use_column_width=True)

        # Conditional Content
        elif page == "Help":
        st.write("Help page")

        # Conditional content
        elif page == "User Guide":
            st.write("""
                **This guide will help you navigate the interface and effectively use the features of this program.**

                ### 1. Getting Started
                To begin using the application:
                - **Open the Streamlit Interface**: Follow the installation instructions in the cloned github repository. This will lead you to running the final program via terminal, which will open browser of choice.
                - **Sign In**: As authentication is enabled, sign in with your credentials. For testing, please use "test" for username, and "456" for password.

                ### 2. Uploading Data on Home Page
                Once logged in:
                - **Upload Your Data on the Home Page**: Use the "Upload" button to select and upload your dataset. The application supports JPG format.

                ### 3. Running Pattern Recognition
                After uploading your data:
                - **Start Processing**: The program will begin running the image recognition program. The application compares OpenCL and CPU for efficient data processing, leveraging your hardware's capabilities.

                ### 4. Viewing Results
                Once the processing is complete:
                - **Access Results**: The results will be displayed on the Home page. This includes:
                    - **Recognition Outputs**: Visual and numerical outputs highlighting the identified patterns in your data.
                    - **Performance Metrics**: Information on the speed and efficiency of the data processing, leveraging OpenCL’s capabilities.

                ### 5. Troubleshooting
                - **Error Handling**: If an error occurs, a message will be displayed with details. Ensure your data is in the correct format and try again.
                - **Support**: For further assistance, contact the support team through the "Help" section.

                ### 6. Logging Out
                - **End Your Session**: When you’re finished, click the "Logout" button to securely end your session.

                **Thank you for using the Pattern Recognition Application!**
                """)


    elif st.session_state['authentication_status'] is False:
        st.error("Username/password is incorrect")


if __name__ == "__main__":
    main()
