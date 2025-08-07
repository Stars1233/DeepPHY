import time
import sys
import os
import json
import re
import argparse
import random

# --- User Configuration ---
# These variables control the starting point of the automation.
TARGET_SENCE_INDEX = 1
TARGET_LEVEL = 1
MAX_STEPS_PER_LEVEL = 3

parser = argparse.ArgumentParser(description="Run the Angry Birds VLM agent.")
parser.add_argument(
    '--model',
    type=str,
    default="api-claude_sonnet4",
    help='The name of the LLM to use (e.g., "api-claude_sonnet4").'
)
args = parser.parse_args()
MODEL_NAME = args.model
print(f"Using model: {MODEL_NAME}")

PROMPT_FORMAT = "VLA"

# --- Path Configuration ---
from deepphy.planner.prompt_strategy_05_angrybirds import *
from deepphy.config import Config
CONFIG_JSON_PATH = 'conf/env_config_05_angrybirds.json'
config = Config()
config.load_env_config(CONFIG_JSON_PATH)

from deepphy.provider.llm.base import *
from deepphy.utils.img_utils import *
from deepphy.utils.gui_utils import *

def parse_and_execute_action(action_str, app_name, level_data):
    """
    Parses the LLM's action string and executes the corresponding GUI action.
    Returns True if the action is terminal for the level (e.g., 'success'),
    and False otherwise (e.g., a 'shoot' action).
    """
    print(f"  - Attempting to execute: {action_str}")

    # Check for terminal actions that end the level
    if "success" in action_str:
        print(f"  - Terminal action '{action_str}' detected. Level will end after this step.")
        return True

    perform_action_swipe(app_name, direction='right')
    shoot_match = re.search(r"shoot\s*\(\s*angle=(-?\d+\.?\d*)\s*,\s*power=(\d+\.?\d*)\s*\)", action_str)
    if shoot_match:
        angle = float(shoot_match.group(1))
        power = float(shoot_match.group(2))

        perform_action_shoot(
            app_name,
            start_point = level_data['slingshot_pos'],
            angle_degrees=angle,
            power_ratio=power
        )

        return False

    # If the action string cannot be parsed into a known action
    print(f"  - [!] Warning: Could not parse or execute action '{action_str}'. Action skipped.")
    return False

def play_level_with_llm(app_name, level_log_dir, settings, ui_map):

    action_history = []
    screenshot_history = []  # 新增：用于存储所有截图的路径

    level_run_log = {
        "level_id": TARGET_LEVEL,
        "scene_index": TARGET_SENCE_INDEX,
        "model_name": MODEL_NAME,
        "status": "failed_max_steps",
        "steps": []
    }

    print(f"config: {config}")

    try:
        scene_data = getattr(config, f'Scene {TARGET_SENCE_INDEX}', None)
        static_level_data = scene_data.get(f'level {TARGET_LEVEL}', {})
    except (KeyError, AttributeError):
        print(f"[!] Error: Scene {TARGET_SENCE_INDEX} or Level {TARGET_LEVEL} data not found in config.")
        level_run_log['status'] = "error_level_data_not_found"
        return level_run_log

    level_complete = False

    perform_action_swipe(app_name, direction='left')
    time.sleep(settings['default_wait_time'])
    current_screenshot_path = take_window_screenshot(app_name, level_log_dir, f"initial.png")
    if not current_screenshot_path:
        print(f"[!] Error: Failed to take initial screenshot for Level {TARGET_LEVEL}.")
        level_run_log['status'] = "failed_screenshot"
        return level_run_log

    screenshot_history.append(current_screenshot_path)  # 新增：将初始截图添加到历史记录

    print(f"\n--- Level {TARGET_LEVEL}, Preparing Initial State ---")

    max_step = static_level_data.get('bird_number', MAX_STEPS_PER_LEVEL)
    print(f"  - Maximum steps allowed for this level: {max_step}")

    for step in range(max_step):
        print(f"\n--- Level {TARGET_LEVEL}, Step {step + 1}/{max_step} ---")

        current_step_level_data = static_level_data.copy()

        # --- Get LLM Action ---
        llm_interaction = get_llm_action(
            screenshot_history,  # 修改：传递整个截图历史记录
            current_step_level_data,
            action_history,
            settings
        )

        raw_llm_response = llm_interaction["raw_response"]

        print(f"  - LLM Suggestion: {raw_llm_response}")
        print("=" * 50)

        if not raw_llm_response:
            print(f"  - [!] Warning: LLM returned an empty response. Skipping this step.")
            raw_llm_response = "PARSE_ERROR"
            continue

        cleaned_response = re.sub(r'```(?:python)?\n(.*?)\n```', r'\1', raw_llm_response, flags=re.DOTALL).strip()
        matches = re.findall(r'\[(.*?)\]', cleaned_response)

        step_log_entry = {
            "step": step + 1,
            "img_path": current_screenshot_path,
            "action_history_before_step": list(action_history),
            "llm_raw_response": raw_llm_response,
            "parsed_action": None,
            "execution_outcome": "not_executed"
        }

        if not matches:
            print(f"  - [!] No valid action found in LLM response: '{raw_llm_response}'.")
            step_log_entry["parsed_action"] = "PARSE_ERROR"
            step_log_entry["execution_outcome"] = "Skipped due to parse error."
            level_run_log["steps"].append(step_log_entry)
            continue

        final_action_str = matches[-1].strip()
        print(f"  - Parsed Action: {final_action_str}")
        action_history.append(final_action_str)
        step_log_entry["parsed_action"] = final_action_str

        # --- Execute the action and handle special bird abilities ---
        is_terminal_action = parse_and_execute_action(final_action_str, app_name, current_step_level_data)

        # Handle special actions for specific birds after a 'shoot' action
        if "shoot" in final_action_str:
            bird_types = static_level_data.get("bird_type", ["red"])
            current_bird_index = len(action_history) - 1
            if current_bird_index < len(bird_types):
                current_bird = bird_types[current_bird_index]
                print(f"  - [Post-Shoot] Current bird is '{current_bird}'. Checking for special action.")

                if current_bird in ["yellow", "blue"]:
                    print(f"  - [Post-Shoot] '{current_bird}' bird detected. Performing screen tap after a short delay.")
                    time.sleep(0.3)
                    perform_action_at_relative_coords(app_name, {'x': 0.4, 'y': 0.8, 'action': 'click'})
                elif current_bird == "black":
                    print(f"  - [Post-Shoot] '{current_bird}' bird detected. Waiting for explosion.")
                    time.sleep(2) # Add extra wait time for the black bird's bomb to detonate

            time.sleep(8)

        if is_terminal_action:
            if "success" in final_action_str:
                print(f"[+] LLM indicated success on step {step + 1}. Level complete.")
                step_log_entry["execution_outcome"] = "success_and_level_complete"
                level_run_log["steps"].append(step_log_entry)
                level_run_log["status"] = "completed"
                level_complete = True
                break

        time.sleep(settings['default_wait_time'])

        print("  - [Post-Action] Waiting for physics to settle...")

        perform_action_swipe(app_name, direction='left')
        time.sleep(settings['default_wait_time'])
        next_screenshot_path = take_window_screenshot(app_name, level_log_dir, f"step_{step + 1:02}.png")

        if not next_screenshot_path:
            print("[!] Failed to take post-action screenshot, cannot continue level.")
            level_run_log['status'] = "failed_screenshot"
            step_log_entry["execution_outcome"] = "action_executed_screenshot_failed"
            level_run_log["steps"].append(step_log_entry)
            break

        print("  - [Post-Action] Pausing game for next analysis...")

        current_screenshot_path = next_screenshot_path
        screenshot_history.append(current_screenshot_path)

        step_log_entry["execution_outcome"] = "action_executed_continue_level"
        level_run_log["steps"].append(step_log_entry)

    # 点击一下快进键
    perform_action_at_relative_coords(app_name, ui_map['skip_animation'])
    time.sleep(settings['default_wait_time'])

    # --- After loop (completion, max steps, or error): Take final screenshot ---
    print(f"\n[+] Taking final screenshot for Level {TARGET_LEVEL}...")
    final_screenshot_filename = f"end.png"
    take_window_screenshot(app_name, level_log_dir, final_screenshot_filename)

    if not level_complete and level_run_log['status'] == "failed_max_steps":
        print(f"[!] Level {TARGET_LEVEL} not completed within {max_step} steps.")

    #  回到菜单页面
    perform_action_at_relative_coords(app_name, ui_map['in_game_functions']['after_level_menu'])
    time.sleep(settings['default_wait_time'])

    return level_run_log

def get_llm_action(screenshot_history_paths, level_data, history, settings):
    """Constructs prompts, gets a raw response from the LLM, and returns the interaction details."""

    if MODEL_NAME == "mock":
        random_angle = round(random.uniform(0, 90.0), 2)
        random_power = round(random.uniform(0.5, 1.0), 2)
        mock_action = f"shoot(angle={random_angle}, power={random_power})"

        mock_response = f"[{mock_action}]"
        print(f"\n[Mock Model] Generating random action: {mock_response}")
        return {
            "raw_response": mock_response,
            "system_prompt": "N/A for mock model",
            "user_prompt": "N/A for mock model"
        }

    system_prompt = system_prompt_angry_birds

    bird_types = level_data.get("bird_type", ["red"]) # Default to one red bird
    current_bird_index = len(history)

    # Determine the current bird and the queue of upcoming birds
    if current_bird_index < len(bird_types):
        current_bird = bird_types[current_bird_index]
        upcoming_birds = bird_types[current_bird_index + 1:]
    else:
        # This case happens if we are on a step beyond the number of birds, which shouldn't happen with correct loop logic
        current_bird = "None"
        upcoming_birds = []

    # --- Dynamically construct the prompt based on available birds ---
    state_analysis_parts = [
        f"*   **Current Bird on Slingshot:** `{current_bird}`",
        f"*   **Remaining Birds in Queue:** `{upcoming_birds if upcoming_birds else 'None'}`"
    ]
    action_instruction_parts = [
        "*   `shoot(angle=X, power=Y)`: Launch the current bird. `X` is an angle in degrees (0 to 90). `Y` is a power level (0.0 to 1.0).",
        "*   `success()`: Call this ONLY if you are certain all pigs have been destroyed and no more shots are needed."
    ]

    state_analysis_prompt = "\n".join(state_analysis_parts)
    action_instructions_prompt = "\n".join(action_instruction_parts)

    history_prompt_text = ""
    if history:
        history_text = "\n".join([f"- Shot {i+1}: `{h}`" for i, h in enumerate(history)])
        history_prompt_text = f"{history_text}"
    else:
        history_prompt_text = "This is the first shot of the level. No actions have been taken yet."

    user_prompt = get_user_prompt_angrybirds(
        state_analysis_prompt=state_analysis_prompt,
        history_prompt_text=history_prompt_text,
        action_instructions_prompt=action_instructions_prompt
    )

    image_descriptions = []

    for i, path in enumerate(screenshot_history_paths[:-1]):
        label = f"This was the game state before Shot {i + 1}. The action taken was `{history[i]}`."
        image_descriptions.append({'path': path, 'label': label})

    # 添加当前状态，即列表中的最后一个
    current_screenshot_path = screenshot_history_paths[-1]
    crop_image_by_cm(current_screenshot_path, crop_top_cm=1.10)
    label = "This is the current game state. Plan your next shot based on this image."
    image_descriptions.append({'path': current_screenshot_path, 'label': label})


    print("=" * 50)
    print("image_descriptions:", image_descriptions)
    print("-" * 50)
    # print("system_prompt:", system_prompt)
    # print("-" * 50)
    # print("user_prompt:", user_prompt)
    # print("=" * 50)
    # import pdb; pdb.set_trace()

    response = get_model_response(image_descriptions, system_prompt, user_prompt, MODEL_NAME)

    print(f"\n[LLM Response] {response}")

    return {
        "raw_response": response,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }

def main():
    """Main execution function to loop through and play levels with the LLM."""
    global TARGET_LEVEL

    print("=" * 50)
    print("--- 'Angry Birds VLM Agent Player ---")
    settings, ui_map, APP_NAME = config.settings, config.ui_map, config.settings['app_name']

    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(settings['tmp_log_dir'], MODEL_NAME, PROMPT_FORMAT, run_timestamp)
    os.makedirs(run_log_dir, exist_ok=True)

    # List to hold all individual level run logs
    all_level_runs_log = []
    # Define the consolidated log path before the loop
    consolidated_log_path = os.path.join(run_log_dir, "run_log.json")
    meta_data = {
        "app_name": APP_NAME,
        "model_name": MODEL_NAME,
        "prompt_format": PROMPT_FORMAT
    }

    all_level_runs_log.append(meta_data)
    with open(consolidated_log_path, 'w', encoding='utf-8') as f:
        json.dump(all_level_runs_log, f, indent=4, ensure_ascii=False)

    print(f"Starting Scene Index: {TARGET_SENCE_INDEX}, Level Index: {TARGET_LEVEL}")
    print(f"Using Model: {MODEL_NAME}")
    print(f"Run logs will be saved in: {run_log_dir}")
    print("=" * 50)

    if get_window_geometry_macos(APP_NAME) is None:
        print(f"[!] Pre-flight check failed: Cannot find window for '{APP_NAME}'. Is it running?")
        sys.exit(1)

    while str(TARGET_LEVEL) in ui_map["levels"]:
        print("\n" + "#" * 50)
        print(f"### Preparing to run Level: {TARGET_LEVEL} ###")
        print("#" * 50)

        print(f"\nSelecting Level {TARGET_LEVEL}...")
        perform_action_at_relative_coords(APP_NAME, ui_map['levels'][str(TARGET_LEVEL)])
        time.sleep(settings['default_wait_time'])

        if TARGET_LEVEL == 1:
            print(f"Skipping animation for Level {TARGET_LEVEL}...")
            perform_action_at_relative_coords(APP_NAME, ui_map['skip_animation'])
            time.sleep(settings['default_wait_time'])

        time.sleep(settings['default_wait_time'])

        level_log_dir = os.path.join(run_log_dir, f"Scene_{TARGET_SENCE_INDEX}_Level_{TARGET_LEVEL}")
        print(f"  - Storing screenshots for this run in: {level_log_dir}")
        os.makedirs(level_log_dir, exist_ok=True)

        print(f"\nStarting LLM-driven gameplay for Level {TARGET_LEVEL}...")


        level_log = play_level_with_llm(APP_NAME, level_log_dir, settings, ui_map)
        all_level_runs_log.append(level_log)


        print("\n" + "=" * 50)
        print(f"Automation attempt complete for: Scene {TARGET_SENCE_INDEX} - Level {TARGET_LEVEL}!")

        # Save the consolidated log file after every level
        with open(consolidated_log_path, 'w', encoding='utf-8') as f:
            json.dump(all_level_runs_log, f, indent=4, ensure_ascii=False)
        print(f"  - Consolidated log updated and saved to: {consolidated_log_path}")
        print("=" * 50)

        TARGET_LEVEL += 1
        if TARGET_SENCE_INDEX == 1 and TARGET_LEVEL > 21: break
        if TARGET_SENCE_INDEX == 2 and TARGET_LEVEL > 13: break

    print(f"\n[INFO] All configured levels have been attempted. Final log is located at {consolidated_log_path}")
    print(f"\n[INFO] Automation finished.")

if __name__ == "__main__":
    main()