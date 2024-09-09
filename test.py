import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import gradio as gr
import time

model = AutoModelForCausalLM.from_pretrained(
    "MILVLG/imp-v1-3b",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("MILVLG/imp-v1-3b", trust_remote_code=True)


USER_PROMPT = ""
SYSTEM_PROMPT = f"""
    A chat between a curious user and an artificial intelligence assistant. The assistant generates helpful, and detailed testcases for software/website testing.
    You are tasked with generating detailed, step-by-step test cases for software functionality based on uploaded images. The user will provide one or more images of a software or website interface. For each image, generate a separate set of test cases following the format below:

    Description: Provide a brief explanation of the functionality being tested, as inferred from the image.

    Pre-conditions: Identify any setup requirements, dependencies, or conditions that must be met before testing can begin (e.g., user logged in, specific data pre-populated, etc.).

    Testing Steps: Outline a clear, numbered sequence of actions that a user would take to test the functionality in the image.

    Expected Result: Specify the expected outcome if the functionality is working as intended.

    Ensure that:

    Test cases are created independently for each image.
    The functionality from each image is fully covered in its own set of test cases.
    Any assumptions you make are clearly stated.
    The focus is on usability, navigation, and feature correctness as demonstrated in the UI of the uploaded images.

    USER: <image>\n{USER_PROMPT}
    ASSISTANT:
    """

image = Image.open("Screenshot 2024-09-07 230747.png")
input_ids = tokenizer(SYSTEM_PROMPT, return_tensors="pt").input_ids
image_tensor = model.image_preprocess(image)

print(f"input : {input_ids.shape}")
print(f"image : {image_tensor.shape}")

output_ids = model.generate(
    input_ids,
    max_new_tokens=300,
    images=image_tensor,
    use_cache=False,
)[0]
print(
    tokenizer.decode(output_ids[input_ids.shape[1] :], skip_special_tokens=True).strip()
)
