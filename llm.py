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


def response(USER_DATA, TOKEN) -> str:
    print(USER_DATA)
    MESSAGE = USER_DATA["text"]
    NUM_FILES = len(USER_DATA["files"])
    FILES = USER_DATA["files"]

    SYSTEM_PROMPT = f"""
    A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed testcase related to image uploaded. You are tasked with generating detailed, step-by-step test cases for software functionality based on uploaded images. The user will provide one or more images of a software or website interface. For each image, generate a separate set of test cases following the format below:

    '''Description: Provide a brief explanation of the functionality being tested, as inferred from the image.
    Pre-conditions: Identify any setup requirements, dependencies, or conditions that must be met before testing can begin (e.g., user logged in, specific data pre-populated, etc.).
    Testing Steps: Outline a clear, numbered sequence of actions that a user would take to test the functionality in the image.
    Expected Result: Specify the expected outcome if the functionality is working as intended.'''

    Ensure that:
    Testcases should be related to validation of data, component interactions, navigation, etc.
    Each testcase should have it's own Description, Pre-conidtions, Testing Steps, Expected Result.

    USER: <image>\n{MESSAGE}
    ASSISTANT:
    """

    RES = generate_answer(FILES, SYSTEM_PROMPT)

    response = f"{RES}."
    return response
    for i in range(len(response)):
        time.sleep(0.025)
        yield response[: i + 1]


def generate_answer(IMAGES: list, SYSTEM_PROMPT) -> str:
    print(len(IMAGES))

    INPUT_IDS = tokenizer(SYSTEM_PROMPT, return_tensors="pt").input_ids

    RESULT = ""
    for EACH_IMG in IMAGES:
        image_path = EACH_IMG["path"]
        image = Image.open(image_path)
        image_tensor = model.image_preprocess(image)

        output_ids = model.generate(
            inputs=INPUT_IDS,
            max_new_tokens=500,
            images=image_tensor,
            use_cache=False,
        )[0]
        CUR_RESULT = tokenizer.decode(
            output_ids[INPUT_IDS.shape[1] :], skip_special_tokens=True
        ).strip()

        RESULT = f"{RESULT} /n/n {CUR_RESULT}"

    return RESULT
