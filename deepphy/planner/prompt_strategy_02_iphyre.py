import re
import numpy as np
import json

from deepphy.config import Config

CONFIG_JSON_PATH='conf/env_config_02_iphyre.json'
config = Config()
config.load_env_config(CONFIG_JSON_PATH)


def _extract_json_actions_from_string(json_raw_string: str) -> list:
    """
    Extracts a list of action dictionaries from a raw string that may contain JSON code blocks or direct JSON.
    This function is designed to handle typical VLM responses that include JSON action parts.
    """
    if not json_raw_string:
        return []

    json_string = None
    # Prefer matching Markdown-style JSON code blocks
    json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', json_raw_string, re.DOTALL)
    if json_match:
        json_string = json_match.group(1)
        # print("  Successfully extracted JSON from Markdown code block.") # Suppressed for sub-function
    else:
        # If no code block is found, try to find the outermost [] directly
        start = json_raw_string.find('[')
        end = json_raw_string.rfind(']')
        if start != -1 and end != -1:
            json_string = json_raw_string[start:end+1]
            # print("  Warning: Markdown code block not found, attempting to extract '[]' content directly.") # Suppressed for sub-function
        else:
            # print("  Parsing failed: No valid JSON array found in the response.") # Suppressed for sub-function
            return []

    if not json_string:
        return []

    try:
        # Remove potential inline comments
        json_string_no_comments = re.sub(r'//.*', '', json_string)
        actions = json.loads(json_string_no_comments)
        if isinstance(actions, list):
            # print(f"  Successfully parsed {len(actions)} actions.") # Suppressed for sub-function
            return actions
        else:
            # print(f"  Parsing failed: The parsed content is not a list, but {type(actions)}.") # Suppressed for sub-function
            return []
    except json.JSONDecodeError as e:
        print(f"  Parsing failed: The extracted string is not valid JSON: {e}")
        print(f"  Extracted string content: {json_string}")
        return []

class PromptStrategy:
    """Base class defining prompt format strategies."""
    def __init__(self, solved_threshold: float):
        self.solved_threshold = solved_threshold

    def generate_prompts(self, attempt_history, image_descriptions, num_eliminable_blocks: int):
        """Generates system and user prompts for the VLM."""
        raise NotImplementedError

    def parse_response(self, vlm_response: str) -> dict:
        """Parses the VLM's text response into a structured dictionary."""
        raise NotImplementedError

    def update_attempt_data(self, base_data: dict, parsed_output: dict) -> dict:
        """Updates attempt data with format-specific fields."""
        return base_data

class VlaStrategy(PromptStrategy):
    """Implements the VLA (Vision-Language-Action) prompt format."""
    def generate_prompts(self, attempt_history, image_descriptions, num_eliminable_blocks):
        system_prompt = (
            f"""You are an expert in 2D physics puzzle solving specializing in trajectory manipulation through strategic block elimination. Your objective is to guide the red ball(s) into the abyss by precisely timing the removal of designated gray blocks (the only eliminable objects).

Game Environment:
1. Core Mechanics:
   - Goal: Get all red balls into the abyss
   - Only gray blocks can be eliminated
   - Static blocks (gray/black) form fixed structures
   - Dynamic blue blocks respond to gravity/physics (e.g. Spring mechanisms, Rigid stick constraints)

Action Requirements:
1. Generate a sequence of elimination actions as a JSON array
2. Each action must specify:
   - 'time': Execution timestamp (0.0-15.0 seconds)
   - 'index': Target block's configuration array position (0-{num_eliminable_blocks - 1})
3. Verify block numbers against the frame before selection
4. {num_eliminable_blocks} gray blocks are currently eligible for elimination

Output Rules:
Absolute adherence to JSON syntax
Time precision to one decimal place recommended

Output Format:
```json
[
    {{"time": 0.5, "index": 2}},
    {{"time": 2.1, "index": 0}}
]
```
"""
        )

        if not attempt_history:
            user_prompt = """
Analyze the initial scene configuration and frame to develop an optimal block elimination sequence. Consider:

1. Ball's initial trajectory
2. Block removal timing consequences
3. Chain reactions from each elimination
4. Physical constraints of the environment

Provide the most efficient solution sequence, the action sequence is:
"""
        else:
            feedback_summary = ""
            for past_attempt in attempt_history:
                status = "FAILED"
                feedback_summary += f"""--- Attempt #{past_attempt['attempt_number']} ---
Actions taken: {json.dumps(past_attempt['parsed_actions'])}.
You FAILED at last attempt.
{status}
"""
            user_prompt = f"""
Previous Attempt Analysis:
{feedback_summary}

Failure Diagnosis:
1. Timing Issues:
   - Early removals causing [specific effects]
   - Late removals resulting in [undesired outcomes]
2. Sequencing Errors:
   - Incorrect block priority
   - Missed chain reaction opportunities
3. Physical Miscalculations:
   - Trajectory inaccuracies
   - Collision mispredictions

Revised Strategy Requirements:
1. Avoid all previously failed approaches
2. Incorporate kinematic insights from failed attempts
3. Optimize for minimal actions
4. Account for newly observed physical behaviors

Propose a refined solution incorporating these lessons, the improved action sequence is:
"""
        return system_prompt, user_prompt

    def parse_response(self, vlm_response: str) -> dict:
        parsed_actions = _extract_json_actions_from_string(vlm_response) # Re-use existing function
        return {"parsed_actions": parsed_actions}

    def update_attempt_data(self, base_data: dict, parsed_output: dict) -> dict:
        base_data['parsed_actions'] = parsed_output.get('parsed_actions', [])
        return base_data

class WmStrategy(PromptStrategy):
    """Implements the WM (World Model) prompt format."""
    def generate_prompts(self, attempt_history, image_descriptions, num_eliminable_blocks):
        system_prompt = (
            f"""You are an expert in 2D physics puzzle solving specializing in trajectory manipulation through strategic block elimination. Your objective is to guide the red ball(s) into the abyss by precisely timing the removal of designated gray blocks (the only eliminable objects).

Game Environment:
1. Core Mechanics:
   - Goal: Get all red balls into the abyss
   - Only gray blocks can be eliminated
   - Static blocks (gray/black) form fixed structures
   - Dynamic blue blocks respond to gravity/physics (e.g. Spring mechanisms, Rigid stick constraints)

Your Task (Two Steps):
1.  **Prediction:** First, predict the physical outcome. Describe the chain of events you expect will occur as a direct result of the action sequence you are about to specify.
2.  **Action:** Second, provide the specific action sequence (time and index) that you based your prediction on.

Action Requirements:
1. Generate a sequence of elimination actions as a JSON array
2. Each action must specify:
   - 'time': Execution timestamp (0.0-15.0 seconds)
   - 'index': Target block's configuration array position (0-{num_eliminable_blocks - 1})
3. Verify block numbers against the frame before selection
4. {num_eliminable_blocks} gray blocks are currently eligible for elimination

Output Rules:
Absolute adherence to JSON syntax
Time precision to one decimal place recommended

Required Output Format:
Your response must strictly follow the format below. Do not include any other words or explanations.
Prediction: [Your brief prediction of the physical interaction.]
Action:
```json
[
    {{"time": 0.5, "index": 2}},
    {{"time": 2.1, "index": 0}}
]
```
Example:
Prediction: The first block will fall and push the red ball to the right, causing it to fall into the abyss. The second block removal is not strictly necessary but ensures the path remains clear.
Action:
```json
[
    {{"time": 0.5, "index": 2}},
    {{"time": 2.1, "index": 0}}
]
```
"""
        )

        if not attempt_history:
            user_prompt = "Analyze the scene, formulate a prediction about how to solve it, and then provide the corresponding action sequence."
        else:
            feedback_summary = ""
            for past_attempt in attempt_history:
                outcome = "The action FAILED."
                feedback_summary += (
                    f"--- Attempt #{past_attempt['attempt_number']} ---\n"
                    f"Your Action: {json.dumps(past_attempt['parsed_actions'])}\n"
                    f"Your Prediction: {past_attempt.get('prediction', 'N/A')}\n"
                    f"Actual Outcome: {outcome}\n\n"
                )

            user_prompt = (
                f"{feedback_summary}\nAnalyze why your previous predictions were wrong by comparing them to the actual outcomes. "
                "Formulate a new prediction and propose a new, better action sequence. "
                "Your response MUST strictly follow the 'Prediction: ... Action: ...' format."
            )
        return system_prompt, user_prompt

    def parse_response(self, vlm_response: str) -> dict:
        prediction_text, action_json_raw_part = _parse_wm_raw_output(vlm_response)
        parsed_actions = _extract_json_actions_from_string(action_json_raw_part) if action_json_raw_part else []
        return {"parsed_actions": parsed_actions, "prediction": prediction_text}

    def update_attempt_data(self, base_data: dict, parsed_output: dict) -> dict:
        base_data['parsed_actions'] = parsed_output.get('parsed_actions', [])
        base_data['prediction'] = parsed_output.get('prediction', 'N/A')
        return base_data

def get_prompt_strategy(format_name: str, solved_threshold: float) -> PromptStrategy:
    """Factory function to get the appropriate prompt strategy object."""
    strategies = {
        "VLA": VlaStrategy,
        "WM": WmStrategy,
    }
    strategy_class = strategies.get(format_name)
    if not strategy_class:
        raise ValueError(f"Unknown prompt format: {format_name}")
    print(f"Using prompt strategy: {format_name}")
    return strategy_class(solved_threshold)


def _parse_wm_raw_output(text_output: str) -> tuple:
    """
    Parses the VLM's raw text output to extract the prediction text and the JSON string part for actions.
    Returns (prediction_text, action_json_string_part)
    """
    prediction_text = "N/A"
    action_json_string_part = None

    # Try to extract Prediction
    # It ends either before "Action:" or at the end of the string
    prediction_match = re.search(r"Prediction\s*:\s*(.*?)(?=\nAction\s*:|\Z)", text_output, re.IGNORECASE | re.DOTALL)
    if prediction_match:
        prediction_text = prediction_match.group(1).strip()

    # Try to extract the part after "Action:"
    action_part_match = re.search(r"Action\s*:\s*(.*)", text_output, re.IGNORECASE | re.DOTALL)
    if action_part_match:
        action_json_string_part = action_part_match.group(1).strip()

    return prediction_text, action_json_string_part