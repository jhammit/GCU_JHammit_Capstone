# GCU_JHammit_Capstone
This is a repository for my Capstone project for my degree with GCU. This project primarily focuses on the comparison between OpenCL and CPU preprocessing for images in combination with ResNet50 and Faster R-CNN.

Briefly describe your project, its purpose, and what you aim to achieve. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

Provide step-by-step instructions on how to install your project. For example:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GCU_JHammit_Capstone.git

2. Open terminal and navigate to project directory, and then avtivate a Virtual Environment appropriate for your operating system:
    ```bash
   cd GCU_JHammit_Capstone.git

3. Install project requirements:
    ```bash
   pip install -r requirements.txt

4. Update filepath for config.yaml file in python file "streamlit-app-resnet50.py"
   - In order to properly test the login feature of the application, the yaml file must be properly called.
   - Please copy the absolute path of the new location of the yaml file, and paste it into the "streamlit-app-resnet50.py" file, in def main under # Load YAML configuration

5. Open terminal and run streamlit command to open streamlit application:
   ```bash
   streamlit run (insert absolute path of "streamlit-app-resnet50.py" here)

6. Once "streamlit run" command has been entered, the resnet50 application will open in preferred browser.

7. Refer to YAML file for login setup, however, login information has been copied below for reference:
   - Username - Password: Test - 456
  
8. If you are attempting to use the Faster R-CNN model for testing, please utilize the pre-built model through Kaggle: https://www.kaggle.com/models/google/faster-rcnn-inception-resnet-v2
   - To use, you will need to download the model file and add it to your program's files.
  

## Usage

This project is intended for personal and educational use only. It is not to be used, modified, or distributed by others without explicit permission from the author.


## Features
- OpenCL vs. CPU image preprocessing
- Streamlit user interface for ResNet50 testing (with OpenCL and CPU image preprocessing comparison)


## Contributing

Contributions are not accepted for this project. If you have suggestions or feedback, please reach out directly.


## License

This project is not licensed for public use. All rights are reserved, and this program is intended for personal and educational use only. Unauthorized use, modification, or distribution is prohibited.


## Acknowledgements

I would like to thank my fellow classmates and GCU professors for their guideance and support through this project.
I would also like to acknowledge the use of Tensorflow, Streamlit, and PyOpenCL, which provided essential functionality for my project.
