import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import time
import json
import base64
import re
import shutil
import datetime
import argparse
from tqdm import tqdm

import random
import math

import numpy as np
from iphyre.games import GAMES
from iphyre.simulator import IPHYRE

from deepphy.config import Config
from deepphy.provider.llm.base import *
from deepphy.utils.img_utils import *
from deepphy.utils.dir_utils import create_main_results_dir
from deepphy.planner.prompt_strategy_02_iphyre import *

CONFIG_JSON_PATH='conf/env_config_02_iphyre.json'
config = Config()
config.load_env_config(CONFIG_JSON_PATH)

def setup_iphyre_environment():
    """Loads the IPHYRE task list."""
    print("Loading IPHYRE task list...")
    tasks = GAMES
    print(f"Successfully loaded {len(tasks)} tasks.")
    return tasks

def get_mock_actions(num_eliminable_blocks: int) -> list:
    """
    Generates a mock list of actions for testing purposes.

    It randomly selects between 90% and 100% of the eliminable blocks and assigns
    them a random elimination time between 0.1 and 15.0 seconds.

    Args:
        num_eliminable_blocks: The total number of blocks that can be eliminated.

    Returns:
        A list of action dictionaries in the format [{'time': t, 'index': i}, ...].
    """
    if num_eliminable_blocks <= 0:
        print("  [MOCK] No eliminable blocks, returning empty action list.")
        return []

    # Step 1: Determine the number of blocks to eliminate (90% to 100%)
    # Use math.ceil to ensure the lower bound is not 0, even with few total blocks
    min_blocks_to_select = math.ceil(0.9 * num_eliminable_blocks)
    num_actions_to_generate = random.randint(min_blocks_to_select, num_eliminable_blocks)

    # Step 2: Randomly select unique block indices from the range [0, num_eliminable_blocks - 1]
    all_possible_indices = range(num_eliminable_blocks)
    selected_indices = random.sample(all_possible_indices, k=num_actions_to_generate)

    # Step 3: Generate an action for each selected index and assign a random time
    mock_actions = []
    for index in selected_indices:
        # Generate a random float between 0.1 and 15.0 for the time
        action_time = random.uniform(0.1, 15.0)

        action = {
            "time": round(action_time, 2),  # Round the time to two decimal places for cleaner output
            "index": index
        }
        mock_actions.append(action)

    print("mock_actions: ", mock_actions)

    print(f"  [MOCK] Generated {len(mock_actions)} mock actions for {num_eliminable_blocks} blocks.")
    return mock_actions

def convert_to_simulation_format(actions: list, env: IPHYRE):
    """Converts JSON actions to the format acceptable by the simulator."""

    print("actions: ", actions)

    action_space = env.get_action_space()[1:]
    print("action_space: ", action_space)
    actions_list = []

    for action in actions:
        if not isinstance(action, dict):
            print(f"Warning: Skipping non-dictionary action {action}")
            continue

        if 'index' not in action or 'time' not in action or not isinstance(action['index'], int) or not isinstance(action['time'], (int, float)):
            print(f"Warning: Skipping malformed action {action}")
            continue

        if action['index'] >= len(action_space):
            print(f"Warning: Action index {action['index']} is out of bounds ({len(action_space)}), correcting.")
            action['index'] = len(action_space) - 1

        actions_list.append([action_space[action['index']][0], action_space[action['index']][1], action['time']])

    actions_list.sort(key=lambda x: x[2])

    for action in actions_list:
        if action[-1] > 0:
            break
        else: action[-1] += 1/config.fps

    print(f"  Converted actions to simulator format: {actions_list}")

    # import pdb; pdb.set_trace()
    return actions_list

def simulate_and_get_reward(actions_list, env):
    """Resets the environment, executes the action sequence, and returns the final reward."""
    env.reset()
    total_step = len(actions_list)
    step, time_count, total_reward = 0, 0, 0

    # Simulation duration is 15 seconds
    while time_count < 15:
        action_taken_this_step = False
        if step < total_step:
            p, t = actions_list[step][0:2], actions_list[step][2]
            if time_count >= t:
                _, reward, done = env.step(p)
                total_reward += reward
                step += 1
                action_taken_this_step = True

        if not action_taken_this_step:
            _, reward, done = env.step([0., 0.]) # Do-nothing action
            total_reward += reward

        time_count += 1 / config.fps
        if done:
            break

    return total_reward


def save_simulation_visuals(env, actions_list, output_dir):
    """Saves the image frames from the simulation sequence."""
    os.makedirs(output_dir, exist_ok=True)
    print(f" Saving simulation sequence frames to: {output_dir}")

    print("actions_list: ", actions_list)
    # import pdb; pdb.set_trace()

    try:
        # collect_seq_data expects a list containing multiple action lists
        env.collect_seq_data(save_path=output_dir, act_lists=[actions_list])

        # Note: output_dir is changed here to the subdirectory where images are actually stored
        actual_frames_dir = os.path.join(output_dir, "0", "images")
        frame_paths = sorted([os.path.join(actual_frames_dir, f) for f in os.listdir(actual_frames_dir) if f.endswith('.png')])

        if not actions_list:
            last_time_value = 1
        else:
            last_time_value = int(actions_list[-1][-1] * config.fps) + 1

        # Crop frames based on the last timestamp in actions_list
        frame_paths =  frame_paths[last_time_value:]
        total_frames = len(frame_paths)

        if total_frames >= config.num_frames_to_select:
            indices = np.linspace(0, total_frames - 1, num=config.num_frames_to_select, dtype=int)
            selected_frames = [frame_paths[i] for i in indices]
        else:
            selected_frames = frame_paths

        # Call the new function to delete unselected frames
        delete_unselected_frames(output_dir, selected_frames)

        return selected_frames

    except Exception as e:
        print(f"  Error: Failed to save sequence data: {e}")
        return []



def process_single_task_with_retries(task_id, task_name, main_results_dir, initial_info, strategy: PromptStrategy):
    """
    Processes a single IPHYRE task with retry logic.
    On each new attempt, it provides the simulation images and results from all previous failed attempts as feedback to the model.
    """
    # --- 1. Task Initialization ---
    task_dir = os.path.join(main_results_dir, f"{task_id:03d}_{task_name}")
    os.makedirs(task_dir, exist_ok=True)
    env = IPHYRE(task_name, fps=config.fps)
    object_config = env.reset()

    # Save initial state image
    initial_img_dir = os.path.join(task_dir, "initial_state")
    os.makedirs(initial_img_dir, exist_ok=True)
    env.collect_initial_data(save_path=initial_img_dir)
    initial_image_path = os.path.join(initial_img_dir, f"{task_name}.png")

    # --- 2. Attempt Loop ---
    is_solved = False
    attempt_history = [] # Store textual history of all attempts
    best_reward = -float('inf')
    final_actions_sim = []

    # Get number of eliminable blocks once per task
    action_space = env.get_action_space()[1:]
    valid_action_space = [pos for pos in action_space if not np.array_equal(pos, [0.0, 0.0])]
    num_eliminable_blocks = len(valid_action_space)

    for attempt_num in range(1, config.max_attempts + 1):
        print(f"\n--- Starting attempt {attempt_num}/{config.max_attempts} for task {task_name} ---")

        # --- 3. Prepare VLM Input (delegated to strategy) ---
        image_descriptions = []
        # Always place the initial state image first
        image_descriptions.append({
            'path': initial_image_path,
            'label': 'This is the initial state of the puzzle.'
        })

        # Add history images, limited to the last config.max_image_history rounds
        # The last element in the 'attempt_history' list is the most recent attempt
        recent_attempts_for_images = attempt_history[-config.max_image_history:]

        print(f"  Adding images from the last {len(recent_attempts_for_images)} failed attempts...")
        for past_attempt in recent_attempts_for_images:
            if 'simulation_frames' in past_attempt and past_attempt['simulation_frames']:
                print(f"    Adding {len(past_attempt['simulation_frames'])} frames from attempt #{past_attempt['attempt_number']}.")
                for j, frame_path in enumerate(past_attempt['simulation_frames']):
                    image_descriptions.append({
                        'path': frame_path,
                        'label': f'Frame {j+1} from failed attempt #{past_attempt["attempt_number"]}'
                    })
            else:
                print(f"    No simulation frames to add from attempt #{past_attempt['attempt_number']}.")

        # Generate prompts using the selected strategy
        # Pass all textual history (attempt_history) to the strategy
        system_prompt, user_prompt = strategy.generate_prompts(attempt_history, image_descriptions, num_eliminable_blocks)

        print(f"  Providing history from {len(attempt_history)} failed attempts for attempt #{attempt_num} (including images from the last {len(recent_attempts_for_images)} attempts).")


        # --- 4. Call VLM and Parse ---
        print(f"  Current number of eliminable blocks: {num_eliminable_blocks}")

        vlm_response = ""
        parsed_output = {}

        model_name = initial_info.get("model")
        if model_name.lower() == "mock":
            print("  [MOCK MODE] Using mock action generator.")
            parsed_actions_mock = get_mock_actions(num_eliminable_blocks)
            vlm_response = f"Mock actions generated for attempt {attempt_num}: " + str(parsed_actions_mock)
            # Simulate parsed_output from strategy based on mock actions
            parsed_output = {"parsed_actions": parsed_actions_mock}
            if isinstance(strategy, WmStrategy):
                parsed_output["prediction"] = ""
        else:
            vlm_response = get_model_response(image_descriptions=image_descriptions, system_prompt=system_prompt, user_prompt=user_prompt, model_name=model_name, qwen_pipe=qwen_pipe)

        parsed_output = strategy.parse_response(vlm_response)
        parsed_actions = parsed_output.get("parsed_actions")

        if not parsed_actions:
            print("  Failed to parse valid actions, proceeding to the next attempt.")
            base_attempt_data = {
                "attempt_number": attempt_num,
                "vlm_response": vlm_response,
                "sim_actions": [], # No actions simulated
                "reward": -1,
                "simulation_frames": []
            }
            # Still update attempt history using strategy to include potential prediction even if actions are invalid
            current_attempt_data = strategy.update_attempt_data(base_attempt_data, parsed_output)
            attempt_history.append(current_attempt_data)
            continue

        actions_for_sim = convert_to_simulation_format(parsed_actions, env)

        # --- 5. Simulate and Evaluate ---
        reward = simulate_and_get_reward(actions_for_sim, env)
        print(f"  (Attempt {attempt_num}) Simulation finished, reward: {reward:.4f}")

        if reward > best_reward:
            best_reward = reward
            final_actions_sim = actions_for_sim

        # Save simulation frames for this attempt
        sim_frames_dir = os.path.join(task_dir, f"attempt_{attempt_num}_frames")
        print("sim_frames_dir: ", sim_frames_dir)
        sim_frames = save_simulation_visuals(env, actions_for_sim, sim_frames_dir)
        print("sim_frames: ", sim_frames)

        # --- 6. Record Current Attempt in History (delegated to strategy) ---
        reward_str = str(reward)
        base_attempt_data = {
            "attempt_number": attempt_num,
            "vlm_response": vlm_response,
            "sim_actions": actions_for_sim,
            "reward": reward_str,
            "simulation_frames": sim_frames
        }
        current_attempt_data = strategy.update_attempt_data(base_attempt_data, parsed_output)
        attempt_history.append(current_attempt_data)

        # --- 7. Check if Solved ---
        if reward >= config.solved_threshold:
            print(f"*** Task {task_name} solved successfully on attempt {attempt_num}! ***\n")
            is_solved = True
            break
        else:
            print(f"  Attempt {attempt_num} failed to solve the task.")
            if attempt_num < config.max_attempts:
                print("  Preparing for the next attempt...")

    # --- 8. Return Final Result ---
    best_reward_str = str(best_reward)
    final_result = {
        "task_id": task_id,
        "task_name": task_name,
        "is_solved": is_solved,
        "total_attempts": len(attempt_history),
        "best_reward": best_reward_stradd,
        "final_actions": final_actions_sim,
        "output_dir": task_dir,
        "attempt_history": attempt_history,
        "notes": f"Task solved in {attempt_history[-1]['attempt_number']} attempts." if is_solved else f"Task not solved after {config.max_attempts} attempts."
    }
    return final_result


def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run IPHYRE Benchmark with a VLM and retry mechanism.")
    parser.add_argument("--model", type=str, default="mock", help="The name of the VLM model to use.")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode to run only a few tasks.")
    parser.add_argument("--start_id", type=int, default=0, help="The starting index of the task ID to process.")
    parser.add_argument("--log_dir_base", type=str, default="tmp_log/02_iphyre/", help="The base directory to store results for all runs.")
    parser.add_argument("--resume_dir", type=str, default=None, help="Specify an existing run directory to resume from.")
    parser.add_argument("--format", type=str, default="VLA", choices=["VLA", "WM"], help="The prompt format to use: VLA (action-only) or WM (world model).") # New parameter
    args = parser.parse_args()

    global qwen_pipe
    qwen_pipe = None

    # --- 2. Setup Environment and Logging ---
    INITIAL_INFO = vars(args)
    all_results = []

    if args.resume_dir:
        main_results_dir = args.resume_dir
        print(f"\nDetected --resume_dir. Attempting to resume from '{main_results_dir}'.")
        summary_file_path = os.path.join(main_results_dir, "summary_results.json")
        completed_tasks = set()

        if os.path.exists(summary_file_path):
            with open(summary_file_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            INITIAL_INFO = all_results[0] # Restore configuration from log
            print(f"Loaded configuration from log file: {INITIAL_INFO}")

            # Check and sync key configurations, especially 'format'
            for key in ["model", "format"]: # Ensure the format is taken from the resume log
                if INITIAL_INFO.get(key) != getattr(args, key):
                    print(f"Warning: Command-line {key} ('{getattr(args, key)}') does not match log file ('{INITIAL_INFO.get(key)}').")
                    setattr(args, key, INITIAL_INFO.get(key))
                    print(f"--> Forcing use of setting from log file: {key} = '{getattr(args, key)}'")

            completed_tasks = {res['task_name'] for res in all_results[1:]}
            print(f"Found {len(completed_tasks)} already processed tasks.")
        else:
            print(f"Warning: summary_results.json not found in '{main_results_dir}'. Starting a new run in this directory.")
            all_results = [INITIAL_INFO]
    else:
        full_log_path = os.path.join(args.log_dir_base, args.model, args.format)
        main_results_dir = create_main_results_dir(full_log_path)
        all_results = [INITIAL_INFO]

    summary_file_path = os.path.join(main_results_dir, "summary_results.json")

    tasks = setup_iphyre_environment()

    if args.resume_dir:
        tasks_to_process = [(i, name) for i, name in enumerate(tasks) if name not in completed_tasks]
    else:
        tasks_to_process = list(enumerate(tasks))[args.start_id:]

    if args.debug:
        tasks_to_process = tasks_to_process[25:26]
        print(f"Debug mode enabled, processing only {len(tasks_to_process)} tasks.")

    if not tasks_to_process:
        print("\nAll tasks are already completed. Nothing to do.")
        return

    print("\n" + "="*50)
    print(f"Starting to process {len(tasks_to_process)} tasks (max {config.max_attempts} attempts each)...")
    print(f"Results will be saved in: {main_results_dir}")
    print("="*50 + "\n")

    try:
        strategy = get_prompt_strategy(INITIAL_INFO["format"], config.solved_threshold)
    except ValueError as e:
        print(f"Error: {e}. Aborting.")
        return

    if INITIAL_INFO["model"].startswith("Qwen"):
        qwen_pipe = initialize_qwen_vlm(INITIAL_INFO["model"])

    # --- 4. Main Loop ---
    for task_id, task_name in tqdm(tasks_to_process, desc="Processing Tasks"):
        print(f"\n--- Starting to process task: {task_id} - {task_name} ---")

        print(f"task_id: {task_id}, task_name: {task_name}, main_results_dir: {main_results_dir}, INITIAL_INFO: {INITIAL_INFO}")
        task_result = process_single_task_with_retries(task_id, task_name, main_results_dir, INITIAL_INFO, strategy)
        all_results.append(task_result)

        try:
            with open(summary_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error: Could not save summary JSON file: {e}")

    print("\n" + "="*50)
    print("All tasks processed!")
    print(f"Final summary results have been successfully saved to: {summary_file_path}")
    print("="*50)

if __name__ == "__main__":
    main()
