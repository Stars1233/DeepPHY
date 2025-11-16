import time
import sys
import os
import json
import ast
import traceback
import re
import argparse
import random

# --- User Configuration ---
# These variables control the starting point of the automation.
TARGET_SESSION_INDEX = 1
TARGET_BOX_INDEX = 1
TARGET_LEVEL = 1
FIXED_CIRCLE_RADIUS = 0.05

MAX_STEPS_PER_LEVEL = 10

parser = argparse.ArgumentParser(description="Run the Cut the Rope VLM agent.")
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
from deepphy.planner.prompt_strategy_06_cutrope import *
from deepphy.config import Config
CONFIG_JSON_PATH = 'conf/env_config_06_cutrope.json'
config = Config()
config.load_env_config(CONFIG_JSON_PATH)

from deepphy.provider.llm.base import *
from deepphy.utils.img_utils import *
from deepphy.utils.gui_utils import *
from deepphy.utils.bubble_detector import detect_bubbles

def parse_and_execute_action(action_str, app_name, level_data):
    """Parses the LLM's action string and executes it."""
    if not action_str:
        print("  - [!] Received empty action string. Skipping.")
        return False

    match = re.match(r"^\s*(\w+)\((.*)\)\s*$", action_str)
    if not match:
        print(f"  - [!] Could not parse action: '{action_str}'. Skipping.")
        return False

    action_name, params_str = match.groups()
    params = {}
    if params_str:
        try:
            params = {k.strip(): ast.literal_eval(v.strip()) for k, v in (item.split('=') for item in params_str.split(','))}
        except Exception as e:
            print(f"  - [!] Could not parse parameters '{params_str}': {e}. Skipping.")
            return False

    print(f"  - Executing Parsed Action: {action_name} with params {params}")
    try:
        if action_name == 'success':
            return True # Signal to end the level
        elif action_name == 'fail':
            print("  - [!] Action 'fail' executed. Ending level with failure.")
            return True
        elif action_name == 'sleep':
            time.sleep(float(params.get('seconds', 1.0)))
        elif action_name == 'cut_pin':
            pin_id = str(int(params.get('id', 0)))
            print("level_data:", level_data)
            print("pin_id:", pin_id)
            coords = level_data.get("Pin", {}).get(pin_id)
            if coords:
                # Check if specific drag coordinates (x1, y1, x2, y2) are provided
                if all(k in coords for k in ('x1', 'y1', 'x2', 'y2')):
                    print("  - Found specific cut coordinates. Performing drag cut.")
                    start_coords = {'x': coords['x1'], 'y': coords['y1']}
                    end_coords = {'x': coords['x2'], 'y': coords['y2']}
                    perform_action_drag(app_name, start_coords, end_coords)
                else:
                    # Fallback to the original circular cut if specific coords are not present
                    print("  - No specific cut coordinates found. Performing circular cut.")
                    perform_cut_circle(app_name, coords, radius=FIXED_CIRCLE_RADIUS)
            else:
                print(f"  - [!] Error: Pin with id {params.get('id')} (key: {pin_id}) not found in config.")
        elif action_name == 'cut_active_pin':
            active_pin_id = str(int(params.get('id', 0)))
            coords = level_data.get("Active Pin", {}).get(active_pin_id)
            if coords:
                perform_cut_circle(app_name, coords, radius=FIXED_CIRCLE_RADIUS)
            else:
                print(f"  - [!] Error: Active Pin with id {params.get('id')} (key: {active_pin_id}) not found in config.")
        elif action_name == 'pop_bubble':
            bubble_id = str(int(params.get('id', 0)))
            coords = level_data.get("Bubble", {}).get(bubble_id)
            if coords:
                # Offset the y coordinate upward to account for bubble rising
                # Default 20% upward offset since bubbles float upward
                y_offset = params.get('y_offset', 0.2)
                adjusted_coords = {
                    'x': coords['x'],
                    'y': max(0.0, coords['y'] - y_offset)  # Ensure y doesn't go below 0
                }
                print(f"  - Popping Bubble {bubble_id} at original {coords}, adjusted to {adjusted_coords} (offset: {y_offset})")
                perform_action_at_relative_coords(app_name, {'action': 'click', **adjusted_coords})
            else:
                print(f"  - [!] Error: Bubble with id {params.get('id')} (key: {bubble_id}) not found in config.")
        elif action_name == 'tap_air_cushion':
            cushion_id = str(int(params.get('id', 0)))
            coords = level_data.get("Air Cushion", {}).get(cushion_id)
            if coords:
                # Make logic robust: handle both integer and list-of-one-integer from LLM
                times_param = params.get('times', 1)
                num_taps = 1
                try:

                    if isinstance(times_param, list) and len(times_param) > 0:
                        num_taps = int(times_param[0])
                    else:
                        num_taps = int(times_param)
                except (ValueError, TypeError):
                    print(f"  - [!] Warning: Could not parse 'times' parameter '{times_param}'. Defaulting to 1 tap.")
                    num_taps = 1

                num_taps = max(1, min(num_taps, 5))
                print(f"    - Tapping Air Cushion {cushion_id} {num_taps} time(s).")

                for i in range(num_taps):
                    perform_action_at_relative_coords(app_name, {'action': 'click', **coords})
                    time.sleep(0.05)
            else:
                 print(f"  - [!] Error: Air Cushion with id {params.get('id')} (key: {cushion_id}) not found in config.")
        elif action_name == 'move_pulley_to':
            pulley_id = str(int(params.get('id', 0)))
            position = params.get('position', 0.5)
            coords = level_data.get("Pulley", {}).get(pulley_id)
            if coords:
                # perform_move_pulley(app_name, coords, position)
                pass
            else:
                print(f"  - [!] Error: Pulley with id {params.get('id')} (key: {pulley_id}) not found in config.")
        elif action_name == 'cut_pulley':
            pulley_id = str(int(params.get('id', 0)))
            coords = level_data.get("Pulley", {}).get(pulley_id)
            if coords:
                # perform_cut_pulley_circle(app_name, coords)
                pass
            else:
                print(f"  - [!] Error: Pulley with id {params.get('id')} (key: {pulley_id}) not found in config.")
        else:
            print(f"  - [!] Unknown action name: '{action_name}'.")
    except Exception as e:
        print(f"  - [!] An error occurred during execution of '{action_str}': {e}")
        traceback.print_exc()
    return False # Signal to continue the loop



def play_level_with_llm(app_name, level_log_dir, settings, ui_map):
    """
    Manages the gameplay for a single level using LLM and returns the log data.
    Logs are no longer written to a file in this function.
    """
    action_history = []

    level_run_log = {
        "level_id": TARGET_LEVEL,
        "box_id": TARGET_BOX_INDEX,
        "session_id": TARGET_SESSION_INDEX,
        "model_name": MODEL_NAME,
        "status": "failed_max_steps",
        "steps": []
    }

    try:
        session_data = getattr(config, f"Session {TARGET_SESSION_INDEX}")
        static_level_data = session_data[f"Box {TARGET_BOX_INDEX}"][f"level {TARGET_LEVEL}"]
    except (KeyError, AttributeError):
        print(f"[!] Error: Could not find coordinate data for Level {TARGET_LEVEL}. Cannot play.")
        level_run_log['status'] = "error_level_data_not_found"
        return level_run_log

    level_complete = False

    current_screenshot_path = take_window_screenshot(app_name, level_log_dir, f"initial.png")
    if not current_screenshot_path:
        print("[!] Failed to take initial paused screenshot, cannot start level.")
        level_run_log['status'] = "failed_screenshot"
        return level_run_log

    print(f"\n--- Level {TARGET_LEVEL}, Preparing Initial State ---")
    print("  - [Initial Setup] Pausing game for first analysis...")
    perform_action_at_relative_coords(app_name, ui_map['in_game_functions']['pause'])
    is_paused = True

    for step in range(MAX_STEPS_PER_LEVEL):
        print(f"\n--- Level {TARGET_LEVEL}, Step {step + 1}/{MAX_STEPS_PER_LEVEL} ---")

        bubble_annotated_path = os.path.join(level_log_dir, f"step_{step+1:02d}_annotated_bubbles.png")
        print(f"  - [State Analysis] Detecting bubbles in '{current_screenshot_path}'...")
        detected_bubbles = detect_bubbles(current_screenshot_path, bubble_annotated_path)
        print(f"  - [State Analysis] Found {len(detected_bubbles)} bubbles. Annotated image saved to '{bubble_annotated_path}'.")

        current_step_level_data = static_level_data.copy()
        if detected_bubbles:
            current_step_level_data['Bubble'] = detected_bubbles
            current_screenshot_path = bubble_annotated_path

        llm_interaction = get_llm_action(
            current_screenshot_path,
            current_step_level_data,
            action_history,
            settings,
            is_first_step=(step == 0)
        )
        raw_llm_response = llm_interaction["raw_response"]
        print(f"  - LLM Suggestion: {raw_llm_response}")
        if not raw_llm_response:
            print("  - [!] LLM returned an empty response. Ending level.")
            raw_llm_response = "No valid response from LLM."
        print("=" * 50)

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
            print(f"  - [!] No valid action found in LLM response: '{raw_llm_response}'. Retrying after a pause.")
            step_log_entry["parsed_action"] = "PARSE_ERROR"
            step_log_entry["execution_outcome"] = "Skipped due to parse error."
            level_run_log["steps"].append(step_log_entry)
            continue

        final_action_str = matches[-1].strip()
        print(f"  - Parsed Action: {final_action_str}")
        action_history.append(final_action_str)
        step_log_entry["parsed_action"] = final_action_str

        # --- Unpause the game if it is paused ---
        # This will now run on every step, including the first one.
        if is_paused:
            print("  - [Action Prep] Resuming game from pause...")
            perform_action_at_relative_coords(app_name, ui_map['in_game_functions']['in_pause_continue'])
            is_paused = False

        # --- Execute the action ---
        # MODIFIED: Pass the updated level data with dynamic bubble info
        if parse_and_execute_action(final_action_str, app_name, current_step_level_data):
            if "success" in final_action_str:
                print(f"[+] LLM indicated success on step {step + 1}. Level complete.")
                step_log_entry["execution_outcome"] = "success_and_level_complete"
                level_run_log["steps"].append(step_log_entry)
                level_run_log["status"] = "completed"
                level_complete = True
                break
            else:
                print(f"[+] LLM indicated failure on step {step + 1}. Ending level with failure.")
                step_log_entry["execution_outcome"] = "fail_and_level_complete"
                level_run_log["steps"].append(step_log_entry)
                level_run_log["status"] = "completed"
                level_complete = True
                break

        print("  - [Post-Action] Waiting for physics to settle...")

        next_screenshot_path = take_window_screenshot(app_name, level_log_dir, f"step_{step + 1:02}.png")

        if not next_screenshot_path:
            print("[!] Failed to take post-action screenshot, cannot continue level.")
            level_run_log['status'] = "failed_screenshot"
            step_log_entry["execution_outcome"] = "action_executed_screenshot_failed"
            level_run_log["steps"].append(step_log_entry)
            break

        print("  - [Post-Action] Pausing game for next analysis...")
        perform_action_at_relative_coords(app_name, ui_map['in_game_functions']['pause'])
        is_paused = True

        current_screenshot_path = next_screenshot_path

        step_log_entry["execution_outcome"] = "action_executed_continue_level"
        level_run_log["steps"].append(step_log_entry)

    # --- After loop (completion, max steps, or error): Take final screenshot ---
    print(f"\n[+] Taking final screenshot for Level {TARGET_LEVEL}...")
    time.sleep(settings['default_wait_time'])
    final_screenshot_filename = f"end.png"
    take_window_screenshot(app_name, level_log_dir, final_screenshot_filename)

    if not level_complete and level_run_log['status'] == "failed_max_steps":
        print(f"[!] Level {TARGET_LEVEL} not completed within {MAX_STEPS_PER_LEVEL} steps.")

    return level_run_log


def get_llm_action(bubble_annotated_path, level_data, history, settings, is_first_step=False):
    """Constructs prompts, gets a raw response from the LLM, or generates a mock action."""

    # --- MOCK MODEL LOGIC ---
    if MODEL_NAME == "mock":
        print("  - [MOCK MODEL] Generating a random action...")
        possible_actions = []

        # Safely get a list of possible actions based on the current level data
        if level_data.get("Pin"):
            for pin_id in level_data.get("Pin", {}).keys():
                action_str = f"cut_pin(id={pin_id})"
                # Only add if this exact action has not been performed before
                if action_str not in history:
                    possible_actions.append(action_str)

        if level_data.get("Active Pin"):
            for pin_id in level_data.get("Active Pin", {}).keys():
                action_str = f"cut_active_pin(id={pin_id})"
                # Only add if this exact action has not been performed before
                if action_str not in history:
                    possible_actions.append(action_str)

        # Per your request, bubble actions are always considered possible, regardless of history.
        if level_data.get("Bubble"):
            for bubble_id in level_data.get("Bubble", {}).keys():
                possible_actions.append(f"pop_bubble(id={bubble_id})")

        if level_data.get("Air Cushion"):
            for cushion_id in level_data.get("Air Cushion", {}).keys():
                action_str = f"tap_air_cushion(id={cushion_id}, times=1)"
                # Check if this cushion has been tapped before
                # This is a simple check; a more complex one could check for any tap on this id.
                if action_str not in history:
                    possible_actions.append(action_str)

        print(f"  - [MOCK MODEL] History: {history}")
        print(f"  - [MOCK MODEL] Possible actions after filtering: {possible_actions}")

        # Pulley actions can be added here if needed in the future
        # if level_data.get("Pulley"):
        #     for pulley_id in level_data.get("Pulley", {}).keys():
        #         possible_actions.append(f"cut_pulley(id={pulley_id})")

        if not possible_actions:
            # If no other actions are possible, fail to prevent an infinite loop or error
            mock_action = "fail()"
        else:
            mock_action = random.choice(possible_actions)

        # The rest of the system expects a specific response format.
        # We wrap the mock action in brackets to mimic the LLM's output format.
        mock_response = f"[{mock_action}]"
        print(f"  - [MOCK MODEL] Selected action: {mock_response}")

        return {
            "raw_response": mock_response,
            "system_prompt": "N/A (Mock Model)",
            "user_prompt": "N/A (Mock Model)"
        }
    # --- END MOCK MODEL LOGIC ---


    # --- Original LLM Logic (no changes below this line in this function) ---
    if TARGET_BOX_INDEX == 4:
        system_prompt = box4_system_prompt
    elif TARGET_BOX_INDEX == 5:
        system_prompt = box5_system_prompt
    else:
        system_prompt = system_prompt_default

    # Use .get() with a default empty dict to avoid KeyErrors
    pins = level_data.get("Pin", {})
    print(f"pins: {pins}")
    active_pins = level_data.get("Active Pin", {})
    print(f"active_pins: {active_pins}")
    bubbles = level_data.get("Bubble", {})
    air_cushions = level_data.get("Air Cushion", {})
    print(f"air_cushions: {air_cushions}")
    pulleys = level_data.get("Pulley", {})
    print(f"pulleys: {pulleys}")

    # import pdb; pdb.set_trace()

    # --- Dynamically construct the prompt based on available game elements ---
    state_analysis_parts = []
    action_instruction_parts = []

    if pins:
        state_analysis_parts.append(f"*   **Pin:** `{len(pins)}` present, with IDs from 1 to `{len(pins)}`")
        action_instruction_parts.append("*   `cut_pin(id=pin_index)` # Effect: Cuts the rope attached to `pin_index`.")

    if active_pins:
        state_analysis_parts.append(f"*   **Active Pin:** `{len(active_pins)}` present, with IDs from 1 to `{len(active_pins)}`")
        action_instruction_parts.append("*   `cut_active_pin(id=active_pin_index)` # Effect: Cuts the rope attached to `active_pin_index`.")

    if bubbles:
        state_analysis_parts.append(f"*   **Bubble:** `{len(bubbles)}` present, with IDs from 1 to `{len(bubbles)}`")
        action_instruction_parts.append("*   `pop_bubble(id=bubble_index)` # Effect: Pops the bubble with the specified `bubble_index`. The candy inside will lose its buoyancy and begin to fall vertically. Only works if there is candy inside the bubble.")

    if air_cushions:
        state_analysis_parts.append(f"*   **Air Cushion:** `{len(air_cushions)}` present, with IDs from 1 to `{len(air_cushions)}`")
        action_instruction_parts.append("*   `tap_air_cushion(id=air_cushion_index, times=[1, 3, 5])` # Effect: Taps the air cushionwith the specified `air_cushion_index` a number of `times`,  to make it release a puff of air, pushing the candy in a specific direction.")

    if pulleys:
        state_analysis_parts.append(f"*   **Pulley:** `{len(pulleys)}` present, with IDs from 1 to `{len(pulleys)}`")
        action_instruction_parts.append("*   `move_pulley_to(id=pulley_index, position=value)` # Effect: Moves the rope anchor on `pulley_index`. The two ends of the pulley's: `position=0` corresponds to the `end_0` position (usually top-left), `position=1` corresponds to the `end_1` position, and `value` is a float between 0 and 1.")
        action_instruction_parts.append("*   `cut_pulley(id=pulley_index)` # Effect: Cuts the rope threaded through the specified Pulley `pulley_index`.")

    # Always include non-conditional actions
    action_instruction_parts.extend([
        "*   `sleep(seconds=x)` # Effect: Waits for `x` seconds, allowing in-game physics to play out. Use this when you need to wait for a physical process (like a swing) to reach a specific state before executing the next action. After calling this, the system will provide a new game state and request the next action after `x` seconds.",
        "*   `success()` # Effect: Declares the mission successful. Call this when you determine the candy is on an inevitable trajectory to enter Om Nom's mouth, requiring no further actions.",
        "*.  `fail()` # Effect: Declares the mission failed. Call this when you determine the candy is lost and no further actions can succeed."
    ])

    state_analysis_prompt = "\n".join(state_analysis_parts)
    action_instructions_prompt = "\n".join(action_instruction_parts)
    # --- End of dynamic construction ---

    history_prompt_text = ""
    if history:
        history_text = "\n".join([f"- Step {i+1}: `{h}`" for i, h in enumerate(history)])
        history_prompt_text = f"{history_text}"
    else:
        history_prompt_text = " This is the first step of the level. No actions have been taken yet."

    user_prompt = get_user_prompt(
        state_analysis_prompt=state_analysis_prompt,
        history_prompt_text=history_prompt_text,
        action_instructions_prompt=action_instructions_prompt
    )

    # --- Construct paths for all three images ---
    image_filename = f"Box_{TARGET_BOX_INDEX:02}_Level_{TARGET_LEVEL:02}_game.png"
    initial_image_path = os.path.join(settings['initial_image_path'], image_filename)
    annotated_image_path = os.path.join(settings['annotated_image_path'], image_filename)

    # --- MODIFIED: The third image is now the one with bubble annotations ---
    if is_first_step:
        current_state_label = "Current game state. This is the initial state of the level. Base your first action on this image."
    else:
        current_state_label = "Current game state captured after the last action. Base your next action on this image."

    image_descriptions = [
        {'path': initial_image_path, 'label': "The initial state of the level (for reference)."},
        {'path': annotated_image_path, 'label': "The initial state of the level with annotations for ALL static interactive elements (pins, cushions, etc.). Use this to identify non-bubble elements by their IDs."},
        {'path': bubble_annotated_path, 'label': current_state_label}
    ]

    # print("=" * 50)
    # print("image_descriptions:", image_descriptions)
    # print("-" * 50)
    # print("system_prompt:", system_prompt)
    # print("-" * 50)
    # print("user_prompt:", user_prompt)
    # print("=" * 50)
    # import pdb; pdb.set_trace()

    response = get_model_response(image_descriptions, system_prompt, user_prompt, MODEL_NAME)

    print(f"\n[LLM Response] {response}")

    # Return a dictionary with all relevant information for logging
    return {
        "raw_response": response,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }


def navigate2level_list(app_name, ui_map, target_box_name, settings):
    print("\n[*] Restarting app for a clean state...")
    restart_mac_app(app_name)
    time.sleep(settings['default_wait_time'] * 3)

    print("\n[STEP 1/5] Clicking 'Play'...")
    perform_action_at_relative_coords(app_name, ui_map['start_play'])
    time.sleep(settings['default_wait_time'] / 2)

    print("\n[STEP 2/5] Selecting session...")
    perform_action_at_relative_coords(app_name, ui_map['select_session_1'])
    time.sleep(settings['default_wait_time'] / 2)

    print(f"\n[STEP 3/5] Selecting box '{target_box_name}'...")
    perform_action_at_relative_coords(app_name, ui_map['boxes'][target_box_name])
    time.sleep(settings['default_wait_time'] / 2)

    print("\n[STEP 4/5] Entering level list...")
    perform_action_at_relative_coords(app_name, ui_map['enter_box_center'])
    time.sleep(settings['default_wait_time'] / 2)


def main():
    """Main execution function to loop through and play levels with the LLM."""
    global TARGET_LEVEL
    global TARGET_BOX_INDEX

    print("=" * 50)
    print("--- 'Cut the Rope' VLM Agent Player ---")
    settings, ui_map, APP_NAME = config.settings, config.ui_map, config.settings['app_name']
    target_box_name = list(ui_map["boxes"].keys())[TARGET_BOX_INDEX - 1]

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

    print(f"Starting Session: {TARGET_SESSION_INDEX}, Box: {target_box_name}, Start Level: {TARGET_LEVEL}")
    print(f"Using Model: {MODEL_NAME}")
    print(f"Run logs will be saved in: {run_log_dir}")
    print("=" * 50)

    if get_window_geometry_macos(APP_NAME) is None:
        print(f"[!] Pre-flight check failed: Cannot find window for '{APP_NAME}'. Is it running?")
        sys.exit(1)

    while str(TARGET_LEVEL) in ui_map["levels"] and TARGET_BOX_INDEX <= 5:
        # print("ui_map['levels']:", ui_map["levels"])
        # import pdb; pdb.set_trace()
        print("\n" + "#" * 50)
        print(f"### Preparing to run Level: {TARGET_LEVEL} ###")
        print("#" * 50)

        try:
            session_data = getattr(config, f"Session {TARGET_SESSION_INDEX}")
            level_data = session_data[f"Box {TARGET_BOX_INDEX}"][f"level {TARGET_LEVEL}"]
            if level_data.get("Pulley"):
                print(f"[!] Level {TARGET_LEVEL} contains a 'Pulley'. Skipping as requested.")

                # Create a log entry for the skipped level
                skipped_level_log = {
                    "level_id": TARGET_LEVEL,
                    "box_id": TARGET_BOX_INDEX,
                    "session_id": TARGET_SESSION_INDEX,
                    "model_name": MODEL_NAME,
                    "status": "skipped_has_pulley",
                    "steps": []
                }
                all_level_runs_log.append(skipped_level_log)

                # Save the updated log
                with open(consolidated_log_path, 'w', encoding='utf-8') as f:
                    json.dump(all_level_runs_log, f, indent=4, ensure_ascii=False)

                TARGET_LEVEL += 1
                if TARGET_LEVEL > 25:
                    TARGET_LEVEL = 1
                    TARGET_BOX_INDEX += 1

                continue

        except (KeyError, AttributeError):
            print(f"[!] Warning: Could not find config data for Level {TARGET_LEVEL} to check for pulleys. Will attempt to play anyway.")

        # --- Standard Navigation to Level ---
        print(f"\n[STEP 0/5] Navigating to Box {TARGET_BOX_INDEX} - {target_box_name}...")
        navigate2level_list(APP_NAME, ui_map, target_box_name, settings)

        print(f"\n[STEP 5/5] Selecting Level {TARGET_LEVEL}...")
        perform_action_at_relative_coords(APP_NAME, ui_map['levels'][str(TARGET_LEVEL)])
        time.sleep(settings['default_wait_time'])

        # --- LLM-driven Gameplay ---
        # Create a subdirectory for this level's screenshots and other assets
        level_log_dir = os.path.join(run_log_dir, f"Box_{TARGET_BOX_INDEX:02}_Level_{TARGET_LEVEL:02}")
        print(f"  - Storing screenshots for this run in: {level_log_dir}")
        os.makedirs(level_log_dir, exist_ok=True)

        print(f"\n[STEP 6/6] Starting LLM-driven gameplay for Level {TARGET_LEVEL}...")

        # Call the function and get the log data back for this level
        # MODIFIED: Pass ui_map to the function
        level_log = play_level_with_llm(APP_NAME, level_log_dir, settings, ui_map)
        all_level_runs_log.append(level_log)

        # --- Post-level & Save Log ---
        print("\n" + "=" * 50)
        print(f"Automation attempt complete for: {target_box_name} - Level {TARGET_LEVEL}!")

        # Save the consolidated log file after every level
        with open(consolidated_log_path, 'w', encoding='utf-8') as f:
            json.dump(all_level_runs_log, f, indent=4, ensure_ascii=False)
        print(f"  - Consolidated log updated and saved to: {consolidated_log_path}")
        print("=" * 50)

        TARGET_LEVEL += 1
        if TARGET_LEVEL > 25:
            TARGET_LEVEL = 1
            TARGET_BOX_INDEX += 1

        if TARGET_BOX_INDEX > 5:
            print(f"\n[!] All configured levels have been attempted. Exiting automation.")
            break
        else:
            target_box_name = list(ui_map["boxes"].keys())[TARGET_BOX_INDEX - 1]

        print(f"\n[INFO] Moving to next level: Box {TARGET_BOX_INDEX}, Level {TARGET_LEVEL}...")

    print(f"\n[INFO] All configured levels have been attempted. Final log is located at {consolidated_log_path}")
    print(f"\n[INFO] Automation finished.")

if __name__ == "__main__":
    main()