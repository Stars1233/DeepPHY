import re
import numpy as np

from deepphy.utils.img_utils import get_normalized_center_coords, convert_radius_size_to_normalized
from deepphy.config import Config

CONFIG_JSON_PATH='conf/env_config_01_phyre.json'
config = Config()
config.load_env_config(CONFIG_JSON_PATH)

def parse_and_convert_vlm_output(text_output: str) -> tuple:
    """"VLA Prompt Format Output Parser and Converter, extracts Cell and Radius from text output and converts to [x, y, r] coordinates."""
    text_output = str(text_output)
    match = re.search(r"Cell\s*[:=\s]*(\d+).*Radius\s*[:=\s]*(\d+)", text_output, re.IGNORECASE | re.DOTALL)
    if not match:
        return None, None, None
    try:
        cell_num = int(match.group(1))
        radius_size = int(match.group(2))
        coords = get_normalized_center_coords(cell_num, config.grid_size)
        if coords is None: return None, cell_num, radius_size
        normalized_r = convert_radius_size_to_normalized(radius_size, config.radius_levels )
        if normalized_r is None: return None, cell_num, radius_size
        action = np.array([coords[0], coords[1], normalized_r])
        return action, cell_num, radius_size
    except (ValueError, IndexError):
        return None, None, None

def parse_wm_vlm_output(text_output: str) -> tuple:
    """ WM Prompt Format Output Parser and Converter, extracts Prediction, Cell and Radius from text output and converts to [x, y, r] coordinates.  """
    text_output = str(text_output)
    prediction_match = re.search(r"Prediction\s*:\s*(.*?)Action\s*:", text_output, re.IGNORECASE | re.DOTALL)
    if not prediction_match:
        prediction_match = re.search(r"Prediction\s*:\s*(.*)", text_output, re.IGNORECASE)
    prediction_text = prediction_match.group(1).strip() if prediction_match else "The model failed to provide a prediction."

    action_match = re.search(r"Action\s*:\s*Cell\s*[:=\s]*(\d+).*Radius\s*[:=\s]*(\d+)", text_output, re.IGNORECASE | re.DOTALL)
    if not action_match:
        return None, None, None, prediction_text
    try:
        cell_num = int(action_match.group(1))
        radius_size = int(action_match.group(2))
        coords = get_normalized_center_coords(cell_num, config.grid_size)
        if coords is None: return None, cell_num, radius_size, prediction_text
        normalized_r = convert_radius_size_to_normalized(radius_size, config.radius_levels)
        if normalized_r is None: return None, cell_num, radius_size, prediction_text
        action = np.array([coords[0], coords[1], normalized_r])
        return action, cell_num, radius_size, prediction_text
    except (ValueError, IndexError):
        return None, None, None, prediction_text

# --- 策略类定义 ---

class PromptStrategy:
    """ Prompt Strategy Interface """
    def __init__(self):
        # 从常量模块获取配置
        self.grid_size = config.grid_size
        self.radius_levels = config.radius_levels

    def generate_prompts(self, attempt_history, image_descriptions):
        """ Generate system and user prompts based on attempt history and image descriptions. """
        raise NotImplementedError

    def parse_response(self, vlm_response: str) -> dict:
        """ Parse the VLM response and return a dictionary with parsed fields. """
        raise NotImplementedError

    def update_attempt_data(self, base_data: dict, parsed_output: dict) -> dict:
        """ Update the base attempt data with additional fields from parsed output. """
        return base_data

class VlaStrategy(PromptStrategy):
    """ VLA Prompt Format Implementation. """
    def generate_prompts(self, attempt_history, image_descriptions):
        system_prompt = (
            f"""You are a premier AI agent, an expert in solving physics-based puzzles. Your task is to analyze a given scene and determine the single optimal placement of a new object to achieve a specific goal.
## Scene Legend:
*   Red Ball: The dynamic ball you will place.
*   Green Ball: The primary dynamic object you must manipulate.
*   Blue Ball: A dynamic target.
*   Purple Ball: A static (immovable) target.
*   Grey Ball: Other dynamic objects.
*   Black Object: Static obstacles.
## Mission Objective:
Strategically place a single red ball to trigger a chain reaction. The successful outcome is achieved when the green ball makes physical contact with either the blue ball or the purple ball.
## Action Parameters:
*   `Cell`: A grid cell number from 1 to {self.grid_size[0] * self.grid_size[1]}.
*   `Radius`: A size level from 1 (smallest) to {self.radius_levels} (largest).
## Placement Constraints:
Your action is only valid if it meets these strict conditions:
*   No Collisions: The red ball cannot overlap with any existing objects.
*   In-Bounds: The red ball must be placed entirely within the puzzle's boundaries.
## Required Output Format:
Your response must be a single line of text, strictly conforming to the format below. Do not include any other words, notes, or explanations.
`Cell: [NUMBER], Radius: [NUMBER]`
Example: `Cell: 13, Radius: 3`"""
        )

        if not attempt_history:
            user_prompt = "Analyze the scene, formulate a prediction about how to solve it, and then provide the corresponding action."
        else:
            feedback_summary = "You have made previous attempts that failed. Here is a summary:\n\n"
            for past_attempt in attempt_history:
                status = "INVALID ACTION" if past_attempt['simulation_status'] in ["INVALID_INPUT", "INVALID_ACTION_FORMAT"] else "FAILED"
                feedback_summary += f"Attempt {past_attempt['attempt_number']}: You chose (Cell: {past_attempt['parsed_cell']}, Radius: {past_attempt['parsed_radius_size']}), which {status}.\n"

            image_note = f"Keyframes from the last {len(image_descriptions)-2} attempts are provided." if len(image_descriptions) > 2 else ""

            user_prompt = (
                f"{feedback_summary}Based on this complete history and the scene images, analyze your mistakes and propose a new, better action. "
                f"{image_note} You MUST propose a NEW action. DO NOT repeat any of the previous actions. "
                f"Your response MUST be ONLY in the format: `Cell: [NUMBER], Radius: [NUMBER]`."
            )
        return system_prompt, user_prompt

    def parse_response(self, vlm_response: str) -> dict:
        action, cell, radius = parse_and_convert_vlm_output(vlm_response)
        return {"action": action, "parsed_cell": cell, "parsed_radius_size": radius}

class WmStrategy(PromptStrategy):
    """ WM Prompt Format Implementation. """
    def generate_prompts(self, attempt_history, image_descriptions):
        system_prompt = (
            f"""You are a premier AI agent, an expert in solving physics-based puzzles. Your task is to analyze a given scene, predict the outcome of an action, and then provide that action.
## Scene Legend:
*   Red Ball: The dynamic ball you will place.
*   Green Ball: The primary dynamic object you must manipulate.
*   Blue Ball: A dynamic target.
*   Purple Ball: A static (immovable) target.
*   Grey Ball: Other dynamic objects.
*   Black Object: Static obstacles.
## Mission Objective:
Strategically place a single red ball to trigger a chain reaction. The successful outcome is achieved when the green ball makes physical contact with either the blue ball or the purple ball.
## Your Task (Two Steps):
1.  **Prediction:** First, predict the physical outcome. Describe the chain of events you expect will occur as a direct result of the action you are about to specify.
2.  **Action:** Second, provide the specific action (Cell and Radius) that you based your prediction on.
## Action Parameters:
*   `Cell`: A grid cell number from 1 to {self.grid_size[0] * self.grid_size[1]}.
*   `Radius`: A size level from 1 (smallest) to {self.radius_levels} (largest).
## Placement Constraints:
Your action is only valid if it meets these strict conditions:
*   No Collisions: The red ball cannot overlap with any existing objects.
*   In-Bounds: The red ball must be placed entirely within the puzzle's boundaries.
## Required Output Format:
Your response must strictly follow the format below. Do not include any other words or explanations.
Prediction: [Your brief prediction of the physical interaction.]
Action: Cell: [NUMBER], Radius: [NUMBER]
Example:
Prediction: The new red ball will drop onto the long black bar, roll to the right, and knock the green ball into the blue target.
Action: Cell: 13, Radius: 3"""
        )

        if not attempt_history:
            user_prompt = "Analyze the scene, formulate a prediction about how to solve it, and then provide the corresponding action."
        else:
            feedback_summary = "You have made previous attempts that failed. Here is a summary of your predictions and the actual outcomes:\n\n"
            for past_attempt in attempt_history:
                outcome = "Your action was INVALID." if past_attempt['simulation_status'] in ["INVALID_INPUT", "INVALID_ACTION_FORMAT"] else "The action FAILED."
                feedback_summary += (
                    f"--- Attempt {past_attempt['attempt_number']} ---\n"
                    f"Your Action: Cell: {past_attempt['parsed_cell']}, Radius: {past_attempt['parsed_radius_size']}\n"
                    f"Your Prediction: {past_attempt.get('prediction', 'N/A')}\n"
                    f"Actual Outcome: {outcome}\n\n"
                )

            image_note = f"Keyframes from the last {len(image_descriptions)-2} attempts are provided as visual evidence." if len(image_descriptions) > 2 else ""

            user_prompt = (
                f"{feedback_summary}Analyze why your previous predictions were wrong by comparing them to the actual outcomes. {image_note} "
                "Formulate a new prediction and propose a new, better action. "
                "Your response MUST strictly follow the 'Prediction: ... Action: ...' format."
            )
        return system_prompt, user_prompt

    def parse_response(self, vlm_response: str) -> dict:
        action, cell, radius, prediction = parse_wm_vlm_output(vlm_response)
        return {"action": action, "parsed_cell": cell, "parsed_radius_size": radius, "prediction": prediction}

    def update_attempt_data(self, base_data: dict, parsed_output: dict) -> dict:
        base_data['prediction'] = parsed_output.get('prediction', 'N/A')
        return base_data

def get_prompt_strategy(format_name: str) -> PromptStrategy:
    """ Factory function to get the appropriate prompt strategy instance based on the format name. """
    strategies = {
        "VLA": VlaStrategy,
        "WM": WmStrategy,
    }
    strategy_class = strategies.get(format_name)
    if not strategy_class:
        raise ValueError(f"Unknown prompt strategy format: {format_name}")
    print(f"Using prompt strategy: {format_name}")
    return strategy_class()
