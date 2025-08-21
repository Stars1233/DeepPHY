import os
import time
from openai import OpenAI

from deepphy.utils.img_utils import encode_image_to_base64

TEMPERATURE = 0.1
SEED = 42
MAX_TOKENS = 4096
MAX_RETRIES = 3

def get_ideatalk_model_response(image_descriptions, system_prompt, user_prompt,
                                      model_name, temperature=TEMPERATURE, seed=SEED, max_tokens=MAX_TOKENS, max_retries=MAX_RETRIES):

    api_url = None
    api_key = None
    api_provider = "openai"

    if 'claude' in model_name:
        if len(image_descriptions) > 20:
            print("Use Claude API, but the number of images exceeds 20, which may cause issues.")
            image_descriptions = image_descriptions[-20:]  # Limit to last 20 images

    if model_name.startswith("bailian-"):
        api_provider = "bailian"
        print("Use BAILIAN API")
        api_url = os.getenv('BAILIAN_API_URL')
        api_key = os.getenv('BAILIAN_API_KEY')
        model_name = model_name.removeprefix("bailian-")

    else:
        print("Use Default OpenAI compatible API")
        api_url = os.getenv('API_URL')
        api_key = os.getenv('API_KEY')
        model_name = model_name.removeprefix("api-")

    print(f"model_name: {model_name}")

    if not api_key or not api_url:
        print(f"Error: {api_provider.upper()} API_KEY or API_URL is not set. Please check your environment variables.")
        return f"Error: {api_provider.upper()} API Key or URL not configured."

    retry_delay = 3

    user_content = []
    for item in image_descriptions:
        # Add text description
        if item.get('label'):
             user_content.append({"type": "text", "text": item['label']})

        # Add image
        base64_image = encode_image_to_base64(item['path'])
        if base64_image:
            user_content.append({
                'type': 'image_url',
                'image_url': {
                    'url': f"data:image/png;base64,{base64_image}"
                }
            })
        else:
            print(f"Warning: Could not encode image {item['path']}, skipping.")

    user_content.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(max_retries):
        try:
            print(f"  {api_provider.capitalize()} API: Requesting action (attempt {attempt + 1}/{max_retries})...")
            client = OpenAI(
                api_key=api_key,
                base_url=api_url,
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"seed": seed}
            )
            print(f"  {api_provider.capitalize()} API: Received response.")

            # --------------------------------------------------------------------

            response_text = response.choices[0].message.content
            return response_text

        except Exception as e:
            print(f"  API Error: An error occurred on attempt {attempt + 1}:")
            print(f"  Error Type: {type(e).__name__}, Error Message: {str(e)}")

            if attempt < max_retries - 1:
                print(f"  Retrying after {retry_delay} seconds...\n")
                time.sleep(retry_delay)
            else:
                print(f"  API: Maximum retries reached. Final failure.")
                return ""