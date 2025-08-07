
import re
import numpy as np
from collections import OrderedDict

from deepphy.config import Config

# Load configuration for grid_size and radius_levels
CONFIG_JSON_PATH='conf/env_config_04_pooltool.json'
config = Config()
config.load_env_config(CONFIG_JSON_PATH)

# Lists of semantic options for VLM prompt
SPEED_OPTIONS_TEXT = ", ".join(config.speed_mapping.keys())
STRIKESPOT_OPTIONS_TEXT = ", ".join(config.strikespot_mapping.keys())

# Regex pattern for parsing VLM output
# Matches "Speed: [SpeedName], Strikespot: [StrikespotName]."
VALID_ACTION_REGEX_PATTERN = r"Speed:\s*(\w+),\s*Strikespot:\s*([\w\s]+)\.?"


# --- Helper parsing functions (mimicking phyre's external parsing functions structure) ---

def _parse_action_components(text_output: str) -> tuple:
    """
    Helper to parse speed and strikespot names from text, then convert to Pooltool parameters.
    Returns (V0, a, b), speed_name, strikespot_name, or None for invalid.
    """
    text_output = str(text_output)

    match = re.search(VALID_ACTION_REGEX_PATTERN, text_output, re.IGNORECASE | re.DOTALL)
    if not match:
        return None, None, None, None # (V0, a, b), speed_name, strikespot_name, raw_output_for_feedback

    speed_name_raw = match.group(1).strip()
    strikespot_name_raw = match.group(2).strip()

    # Find the exact key from the mappings (case-insensitive search)
    found_speed_key = next((key for key in config.speed_mapping.keys() if key.lower() == speed_name_raw.lower()), None)
    found_strikespot_key = next((key for key in config.strikespot_mapping.keys() if key.lower() == strikespot_name_raw.lower()), None)

    if not found_speed_key or not found_strikespot_key:
        return None, None, None, None

    V0 = config.speed_mapping[found_speed_key]
    strikespot_params = config.strikespot_mapping[found_strikespot_key]
    a = strikespot_params["a"]
    b = strikespot_params["b"]

    action_params = (V0, a, b)
    return action_params, found_speed_key, found_strikespot_key, text_output


def parse_vla_vlm_output(text_output: str) -> dict:
    """(For VLA strategy) Parses VLM output for action parameters."""
    action, speed_name, strikespot_name, raw_output = _parse_action_components(text_output)
    return {
        "action": action,
        "parsed_speed_name": speed_name,
        "parsed_strikespot_name": strikespot_name,
        "raw_vlm_output": raw_output
    }


def parse_wm_vlm_output(text_output: str) -> dict:
    """(For WM strategy) Parses VLM output containing 'Prediction' and 'Action' for pooltool."""
    text_output = str(text_output)

    # Extract Prediction
    # Use a lookahead assertion to stop at "Action:" or end of string
    prediction_match = re.search(r"Prediction\s*:\s*(.*?)(?=\nAction\s*:|\Z)", text_output, re.IGNORECASE | re.DOTALL)
    prediction_text = prediction_match.group(1).strip() if prediction_match else "The model failed to provide a prediction."

    # Extract Action using the helper
    action, speed_name, strikespot_name, _ = _parse_action_components(text_output) # _ is for raw_output which is already text_output

    return {
        "action": action,
        "parsed_speed_name": speed_name,
        "parsed_strikespot_name": strikespot_name,
        "prediction": prediction_text,
        "raw_vlm_output": text_output # Store raw output for debugging and full context
    }


# --- Strategy Classes Definition ---

class PromptStrategy:
    """Base class for prompt formatting strategies."""
    def __init__(self):
        self.speed_options_text = SPEED_OPTIONS_TEXT
        self.strikespot_options_text = STRIKESPOT_OPTIONS_TEXT

    def generate_prompts(self, attempt_history: list, image_descriptions: list) -> tuple[str, str]:
        """Generates system and user prompts for the VLM."""
        raise NotImplementedError

    def parse_response(self, vlm_response: str) -> dict:
        """Parses VLM's text response into a structured dictionary."""
        raise NotImplementedError

    def update_attempt_data(self, base_data: dict, parsed_output: dict) -> dict:
        """Updates attempt data with format-specific fields."""
        return base_data

class VlaStrategy(PromptStrategy):
    """Implements the VLA (Vision-Language-Action) prompt format."""
    def generate_prompts(self, attempt_history, image_descriptions):
        system_prompt = (
            f"""You are an expert AI agent specializing in billiards game strategy. Your goal is to guide a player in a 9-ball pool game.

## Game Objective:
Your ultimate objective is to pocket the number 9-ball. In each turn, you will pocket the number 9-ball by hitting the lowest numbered ball. Your task is to provide the optimal cue ball strike parameters to achieve this.
**Crucially, avoid potting the cue ball (white ball). Potting the cue ball is a foul and will negatively impact the game progress.**

## Current Table State:
Image (Current Table State): This is the pool table now. All potted balls from previous shots have been removed. If the cue ball was potted, it has been repositioned. Analyze this image to determine your next move.

## Action Parameters:
You must choose the cue ball's speed and strike spot from the following precise actions.

Available Speed Options: {self.speed_options_text}
Available Strikespot Options: {self.strikespot_options_text}

## Strikespot Effects:
-   **Top Spin (High Cue):** Hitting the cue ball above its center. Makes the cue ball continue forward after contact with an object ball, also known as "follow" or "roll".
-   **Bottom Spin (Low Cue):** Hitting the cue ball below its center. Makes the cue ball spin backward, causing it to rebound or "draw" after hitting an object ball.
-   **Left Spin (Left English):** Hitting the cue ball on its left side. Causes the cue ball to curve or deflect to the left after contact.
-   **Right Spin (Right English):** Hitting the cue ball on its right side. Causes the cue ball to curve or deflect to the right after contact.

## Required Output Format:
Your response must be a single line of text, strictly conforming to the format below. Do not include any other words, notes, or explanations.

`Speed: [Speed Name], Strikespot: [Strikespot Name].`
            """
        )

        # --- MODIFICATION START ---
        # The user requested that the VLM not be passed previous actions,
        # and only receive the current table image.
        user_prompt = "Analyze the scene and determine the best way to pocket the 9-ball into any pocket while avoiding potting the cue ball. Consider the cue ball's position for good follow-up. Provide the optimal speed and strikespot. Then provide the corresponding action that match the format."
        # --- MODIFICATION END ---
        return system_prompt, user_prompt

    def parse_response(self, vlm_response: str) -> dict:
        return parse_vla_vlm_output(vlm_response)


class WmStrategy(PromptStrategy):
    """Implements the WM (World Model) prompt format."""
    def generate_prompts(self, attempt_history, image_descriptions):
        system_prompt = (
            f"""You are an expert AI agent specializing in billiards game strategy. Your goal is to guide a player in a 9-ball pool game.

## Game Objective:
Your ultimate objective is to pocket the number 9-ball. In each turn, you will pocket the number 9-ball by hitting the lowest numbered ball. Your task is to provide the optimal cue ball strike parameters to achieve this.
**Crucially, avoid potting the cue ball (white ball). Potting the cue ball is a foul and will negatively impact the game progress.**

## Current Table State:
Image (Current Table State): This is the pool table now. All potted balls from previous shots have been removed. If the cue ball was potted, it has been repositioned. Analyze this image to determine your next move.

## Your Task (Two Steps):
1.  **Prediction:** First, predict the physical outcome. Describe the chain of events you expect will occur as a direct result of the action you are about to specify.
2.  **Action:** Second, provide the specific action (Speed and Strikespot) that you based your prediction on.

## Action Parameters:
You must choose the cue ball's speed and strike spot from the following precise actions.

Available Speed Options: {self.speed_options_text}
Available Strikespot Options: {self.strikespot_options_text}

## Strikespot Effects:
-   **Top Spin (High Cue):** Hitting the cue ball above its center. Makes the cue ball continue forward after contact with an object ball, also known as "follow" or "roll".
-   **Bottom Spin (Low Cue):** Hitting the cue ball below its center. Makes the cue ball spin backward, causing it to rebound or "draw" after hitting an object ball.
-   **Left Spin (Left English):** Hitting the cue ball on its left side. Causes the cue ball to curve or deflect to the left after contact.
-   **Right Spin (Right English):** Hitting the cue ball on its right side. Causes the cue ball to curve or deflect to the right after contact.

## Required Output Format:
Your response must strictly follow the format below. Do not include any other words or explanations.

Prediction: [Your brief prediction of the physical interaction.]
Action: Speed: [Speed Name], Strikespot: [Strikespot Name].
            """
        )

        # --- MODIFICATION START ---
        # The user requested that the VLM not be passed previous actions,
        # and only receive the current table image.
        user_prompt = "Analyze the scene and determine the best way to pocket the 9-ball into any pocket while avoiding potting the cue ball. Consider the cue ball's position for good follow-up. Provide the optimal speed and strikespot. Then provide the corresponding action that match the format."
        # --- MODIFICATION END ---
        return system_prompt, user_prompt

    def parse_response(self, vlm_response: str) -> dict:
        return parse_wm_vlm_output(vlm_response)

    def update_attempt_data(self, base_data: dict, parsed_output: dict) -> dict:
        # Add the 'prediction' field from parsed_output to the attempt history
        base_data['prediction'] = parsed_output.get('prediction', 'N/A')
        return base_data

def get_prompt_strategy(format_name: str) -> PromptStrategy:
    """Factory function to get the appropriate prompt strategy object."""
    strategies = {
        "VLA": VlaStrategy,
        "WM": WmStrategy,
    }
    strategy_class = strategies.get(format_name)
    if not strategy_class:
        raise ValueError(f"Unknown prompt format: {format_name}")
    print(f"Using prompt strategy: {format_name}")
    return strategy_class()