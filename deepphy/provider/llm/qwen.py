import os
import time
import json
from transformers import pipeline
import torch

MAX_RETRIES = 3
TEMPERATURE = 0.1

def initialize_qwen_vlm(qwen_model_name):

    if qwen_model_name == "Qwen2.5-VL-3B-Instruct":
        QWEN_MODEL_PATH = os.getenv('QWEN_MODEL_3B_PATH')
    if qwen_model_name == "Qwen2.5-VL-7B-Instruct":
        QWEN_MODEL_PATH = os.getenv('QWEN_MODEL_7B_PATH')

    print(f"Initialing Qwen-VL pipeline: {QWEN_MODEL_PATH}")
    qwen_pipe = pipeline(
            model=QWEN_MODEL_PATH,
            task="image-text-to-text",
            model_kwargs={"torch_dtype": torch.float16},
            trust_remote_code=True
        )
    print(f"Qwen-VL pipeline loaded from {QWEN_MODEL_PATH}")

    return qwen_pipe

def get_qwen_model_response(qwen_pipe, image_descriptions, system_prompt, user_prompt, max_retries=MAX_RETRIES):

    user_content = []
    for item in image_descriptions:
        # Add descriptive text
        user_content.append({"type": "text", "text": item['label']})
        # Add corresponding image
        user_content.append({"type": "image", "url": item['path']})

    # Finally, add the user's core instruction
    user_content.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(max_retries):
        try:
            print(f"  Qwen-VL: Requesting action (attempt {attempt + 1}/{max_retries})...")
            # To use temperature, do_sample must be True.
            # You can adjust the temperature value (e.g., 0.7) as needed.
            response = qwen_pipe(
                messages,
                max_new_tokens=200,
                do_sample=True, # Set to True to enable sampling
                temperature=TEMPERATURE, # Add the temperature parameter here
                return_full_text=False
            )
            response_text = response[0]["generated_text"]
            return response_text
        except Exception as e:
            print(f"  Qwen-VL Error: An unexpected error occurred: {e}")
        if attempt < max_retries - 1:
            print("  Qwen-VL: Retrying...")
            time.sleep(2) # Slightly increase wait time

    print("  Qwen-VL: Failed to get a valid response after multiple retries.")
    return ""