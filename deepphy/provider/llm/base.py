from .openai import get_ideatalk_model_response
from .qwen import get_qwen_model_response, initialize_qwen_vlm

def get_model_response(image_descriptions, system_prompt, user_prompt, model_name, qwen_pipe=None):

    print(f"Geting {model_name} response ... ")

    # print("base image_descriptions: ", image_descriptions)
    # print("base system_prompt: ", system_prompt)
    # print("base user_prompt: ", user_prompt)

    if model_name.startswith("Qwen"):
        vlm_response = get_qwen_model_response(qwen_pipe, image_descriptions, system_prompt, user_prompt)
    else:
        vlm_response = get_ideatalk_model_response(image_descriptions, system_prompt, user_prompt, model_name)

    return vlm_response
