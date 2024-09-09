# Multimodal LLM Project

This project implements a multimodal Large Language Model (LLM) using Gradio for the user interface.

## Colab
Mostly everything is already setuped in ipynb file, to use with gradio UI. Use it with Google Colab for GPU. 
But I have use VSCode here.

## Prerequisites

- [Python](https://www.python.org/) 3.7+
- [pip](https://pypi.org/project/pip/) (Python package manager)

## Installation

```

1. Create a virtual environment (optional but recommended):

    ``
    python -m venv venv
    source venv/bin/activate
    ``

2. Install the required packages:

    ``
    pip install -r requirements.txt
    ``

## Usage

1. Run the main script:
``
python app.py
``

2. Open your web browser and go to the URL provided in the terminal (usually `http://127.0.0.1:7860`).

3. Use the Gradio interface to interact with the multimodal LLM:

- Upload images or enter text as input
- Click the "Submit" button to generate results
- View the model's output in the interface
#Mixed Approach Prompt
The tool utilizes a mixed approach to prompt the model effectively, combining the visual information from the UI-Detector with contextual text to generate accurate and useful test cases.

#How It Works
Image Upload: Users upload screenshots through the Gradio interface.
UI Detection: The UI-Detector model processes the images to detect and highlight UI elements.
Image Preprocessing: Screenshots are preprocessed into a format suitable for the MLLM.
Test Case Generation: The OpenGVLab/InternVL2-2B model, guided by a mixed-approach prompt, generates detailed test cases based on the processed images and context.

## Troubleshooting

If you encounter any issues, please check the following:

- Ensure all dependencies are correctly installed
- Verify that you have the necessary GPU drivers if using GPU acceleration
- Check the console output for any error messages
