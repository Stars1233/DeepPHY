import os
import re
import time
import random
import json
import base64
import argparse
import shutil

import numpy as np
from tqdm import tqdm
import phyre
import matplotlib.pyplot as plt

from deepphy.config import Config
from deepphy.provider.llm.base import get_model_response, initialize_qwen_vlm
from deepphy.utils.img_utils import *
from deepphy.utils.dir_utils import create_main_results_dir
from deepphy.planner.prompt_strategy_01_phyre import *

CONFIG_JSON_PATH='conf/env_config_01_phyre.json'
config = Config()
config.load_env_config(CONFIG_JSON_PATH)

def setup_phyre_environment(eval_setup='ball_within_template', fold_id=0, eval_type="train"):
    """Loads the specified PHYRE evaluation setup and initializes the simulator."""
    try:
        print(f"Loading evaluation setup '{eval_setup}' (Fold {fold_id})...")
        train_tasks, _, test_tasks = phyre.get_fold(eval_setup, fold_id)
        eval_tasks = train_tasks if eval_type == "train" else test_tasks
        print(f"Successfully loaded {len(eval_tasks)} {eval_type} tasks.")
        print("Initializing simulator...")
        action_tier = phyre.eval_setup_to_action_tier(eval_setup)
        simulator = phyre.initialize_simulator(eval_tasks, action_tier)
        print("Simulator initialized.")
        return eval_tasks, simulator
    except Exception as e:
        print(f"Error: Could not set up PHYRE environment: {e}")
        return None, None

def get_action_from_vlm_mock() -> str:
    """Provides a mock VLM response for testing purposes."""
    time.sleep(0.1)

    if random.random() < 0.9:
        cell = random.randint(1, config.grid_size[0] * config.grid_size[1])
        radius = random.randint(1, config.radius_levels)
        response = f"Cell: {cell}, Radius: {radius}"
        print(f"  (Mock) VLM generated discrete action: '{response}'")
        return response
    else:
        error_response = "Sorry, I could not determine the best placement from the grid."
        print(f"  (Mock) VLM failed to generate a valid action: '{error_response}'")
        return error_response

def parse_and_convert_vlm_output(text_output: str) -> tuple:
    """(For VLA strategy) Parses "Cell" and "Radius" from VLM output and converts to [x, y, r] coordinates."""
    text_output = str(text_output)
    match = re.search(r"Cell\s*[:=\s]*(\d+).*Radius\s*[:=\s]*(\d+)", text_output, re.IGNORECASE | re.DOTALL)
    if not match:
        return None, None, None
    try:
        cell_num = int(match.group(1))
        radius_size = int(match.group(2))
        coords = get_normalized_center_coords(cell_num, config.grid_size)
        if coords is None: return None, cell_num, radius_size
        normalized_r = convert_radius_size_to_normalized(radius_size, RADIUS_LEVELS)
        if normalized_r is None: return None, cell_num, radius_size
        action = np.array([coords[0], coords[1], normalized_r])
        return action, cell_num, radius_size
    except (ValueError, IndexError):
        return None, None, None

def parse_wm_vlm_output(text_output: str) -> tuple:
    """(For WM strategy) Parses VLM output containing "Prediction" and "Action"."""
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
        normalized_r = convert_radius_size_to_normalized(radius_size, RADIUS_LEVELS)
        if normalized_r is None: return None, cell_num, radius_size, prediction_text
        action = np.array([coords[0], coords[1], normalized_r])
        return action, cell_num, radius_size, prediction_text
    except (ValueError, IndexError):
        return None, None, None, prediction_text

def process_single_task_with_retries(seed, simulator, main_results_dir, INITIAL_INFO, strategy: PromptStrategy):
    """Processes a single PHYRE task with a pluggable prompt strategy and retry logic."""
    # --- 1. Task Initialization ---
    task_subdir_name = seed.replace(':', '-')
    task_dir = os.path.join(main_results_dir, task_subdir_name)
    os.makedirs(task_dir, exist_ok=True)
    task_index = simulator.task_ids.index(seed)
    initial_scene = simulator.initial_scenes[task_index]
    initial_image_path = os.path.join(task_dir, "initial_scene.png")
    plt.imsave(initial_image_path, phyre.observations_to_float_rgb(initial_scene))
    gridded_image_path = add_grid_with_matplotlib(initial_image_path, config.grid_size)
    if not gridded_image_path:
        return {"seed": seed, "is_solved": False, "notes": "Failed to create gridded image."}

    # --- 2. Attempt Loop ---
    is_solved = False
    attempt_history = []

    for attempt_num in range(1, config.max_attempts + 1):
        print(f"\n--- Starting Attempt {attempt_num}/{config.max_attempts } for Task {seed} (Format: {INITIAL_INFO.get('format')}) ---")

        # --- 3. Prepare VLM Input (delegated to strategy) ---
        image_descriptions = [
            {'path': initial_image_path, 'label': "Image 1 (Initial Scene)"},
            {'path': gridded_image_path, 'label': f"Image 2 (Gridded Scene, {config.grid_size[0]}x{config.grid_size[1]})"}
        ]

        # MODIFIED: Only pass images from the most recent MAX_IMAGE_HISTORY attempts
        img_counter = 3
        # Get history for recent N attempts that need images
        recent_attempts_for_images = attempt_history[-config.max_image_history:]
        for past_attempt in recent_attempts_for_images:
            for keyframe in past_attempt.get('simulation_keyframes', []):
                image_descriptions.append({
                    'path': keyframe['path'],
                    'label': f"Image {img_counter} (Result of Attempt {past_attempt['attempt_number']}: {keyframe['label']})"
                })
                img_counter += 1

        # generate_prompts receives the full text history, but the image list is filtered
        system_prompt, user_prompt = strategy.generate_prompts(attempt_history, image_descriptions)

        # --- 4. Call VLM and Simulate ---
        print(f"Preparing to call VLM for attempt {attempt_num} with {len(image_descriptions)} images.")
        vlm_response = ""
        model_name = INITIAL_INFO.get("model")

        if model_name.lower() == "mock":
            vlm_response = get_action_from_vlm_mock()
        else:
            vlm_response = get_model_response(image_descriptions=image_descriptions, system_prompt=system_prompt, user_prompt=user_prompt, model_name=model_name, qwen_pipe=qwen_pipe)

        print("vlm_response: ", vlm_response)
        parsed_output = strategy.parse_response(vlm_response)
        print("parsed_output: ", parsed_output)
        action = parsed_output.get("action")
        print("action: ", action)

        simulation_status_name = "INVALID_ACTION_FORMAT"
        simulation_results = {"gif_path": None, "keyframes": []}

        if action is not None:
            simulation = simulator.simulate_action(task_index, action, need_images=True)
            simulation_status_name = simulation.status.name
            print(f"  Action status: {simulation_status_name}")
            file_prefix = f"attempt_{attempt_num}"
            simulation_results = save_simulation_images_and_get_keyframes(simulation, task_dir, file_prefix)
            is_solved = simulation.status.is_solved()
        else:
            print("  Failed to get a valid action from VLM.\n")

        # --- 5. Record Current Attempt to History ---
        base_attempt_data = {
            "attempt_number": attempt_num, "vlm_response": vlm_response,
            "parsed_cell": parsed_output.get("parsed_cell"),
            "parsed_radius_size": parsed_output.get("parsed_radius_size"),
            "parsed_action": action.tolist() if action is not None else None,
            "simulation_status": simulation_status_name,
            "simulation_keyframes": simulation_results["keyframes"],
            "is_solved_in_this_attempt": is_solved,
        }
        current_attempt_data = strategy.update_attempt_data(base_attempt_data, parsed_output)
        attempt_history.append(current_attempt_data)

        # --- 6. Check for Solution or Abort ---
        if is_solved:
            print(f"*** Task {seed} solved on attempt {attempt_num}! ***\n")
            break
        elif action is None:
            print(f"--- Task {seed} aborted after attempt {attempt_num} due to invalid VLM output. ---")

    # --- 7. Return Final Result ---
    final_attempt_count = attempt_num if 'attempt_num' in locals() else 0
    return {
        "seed": seed, "is_solved": is_solved, "total_attempts": final_attempt_count,
        "output_dir": task_dir, "attempt_history": attempt_history,
        "notes": f"Task solved after {final_attempt_count} attempts." if is_solved else f"Task not solved after {final_attempt_count} attempts."
    }


def main():
    parser = argparse.ArgumentParser(description="Run the PHYRE VLM agent with configurable settings.")
    parser.add_argument("--model", type=str, default="mock", help="The VLM model to use for solving puzzles.")
    parser.add_argument("--eval_setup", type=str, default="ball_within_template", choices=["ball_cross_template", "ball_within_template"], help="The PHYRE evaluation setup to use.")
    parser.add_argument("--format", type=str, default="VLA", choices=["VLA", "WM"], help="The prompt format to use: VLA (action-only) or WM (world model).")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode (processes only a few tasks).")
    parser.add_argument("--start_id", type=int, default=0, help="The starting index of the task ID to process.")
    parser.add_argument("--reuse_log_dir", type=str, default=None, help="Specify an existing log directory to resume a run.")
    parser.add_argument("--eval_type", type=str, default="test", choices=["train", "test"], help="The evaluation type.")
    parser.add_argument("--LOG", type=str, default="tmp_log/01_phyre/", help="Prefix for the log directory.")
    args = parser.parse_args()

    global qwen_pipe
    qwen_pipe = None

    # --- 2. Initialization and Resumption Logic ---
    INITIAL_INFO_from_args = {k: v for k, v in vars(args).items()}
    print(f"Running with the following configuration: {INITIAL_INFO_from_args}")

    LOG_DIR = INITIAL_INFO_from_args["LOG"]
    all_results = []
    tasks_to_process = []

    if args.reuse_log_dir:
        main_results_dir = args.reuse_log_dir
        print(f"\n--reuse_log_dir detected. Attempting to resume run from '{main_results_dir}'.")
        os.makedirs(main_results_dir, exist_ok=True)
        summary_file_path = os.path.join(main_results_dir, "summary_results.json")
        completed_seeds = set()

        if os.path.exists(summary_file_path):
            try:
                with open(summary_file_path, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
                if not all_results: raise ValueError("Log file is empty.")
                INITIAL_INFO = all_results[0]
                print(f"Successfully loaded log file. Using configuration from file: {INITIAL_INFO}")

                # Check and synchronize critical configurations
                for key in ["eval_setup", "model", "eval_type", "format"]:
                    if INITIAL_INFO.get(key) != getattr(args, key):
                        print(f"Warning: Command-line {key} ('{getattr(args, key)}') differs from log file ('{INITIAL_INFO.get(key)}').")
                        setattr(args, key, INITIAL_INFO.get(key))
                        print(f"--> Forcing use of setting from log file: {key} = '{getattr(args, key)}'")

                for result in all_results[1:]:
                    if "seed" in result: completed_seeds.add(result["seed"])
                print(f"Found {len(completed_seeds)} already processed tasks.")
            except Exception as e:
                print(f"Error: Could not parse '{summary_file_path}' ({e}). Starting as a new run in the specified directory.")
                INITIAL_INFO = INITIAL_INFO_from_args
                all_results = [INITIAL_INFO]
        else:
            print(f"Warning: summary_results.json not found in '{main_results_dir}'. Starting as a new run.")
            INITIAL_INFO = INITIAL_INFO_from_args
            all_results = [INITIAL_INFO]
    else:
        INITIAL_INFO = INITIAL_INFO_from_args
        model = INITIAL_INFO.get("model")
        eval_setup = INITIAL_INFO.get("eval_setup")
        eval_type = INITIAL_INFO.get("eval_type")
        format = INITIAL_INFO.get("format")
        full_log_path = os.path.join(LOG_DIR, model, eval_setup, eval_type, format)
        main_results_dir = create_main_results_dir(log_dir=full_log_path)
        all_results = [INITIAL_INFO]

    test_tasks, simulator = setup_phyre_environment(eval_setup=INITIAL_INFO.get("eval_setup"), eval_type=INITIAL_INFO.get("eval_type"))
    if not test_tasks or not simulator:
        print("Cannot continue, environment setup failed.")
        return

    if args.reuse_log_dir: # Filter tasks only in reuse mode
        tasks_to_process = [seed for seed in test_tasks if seed not in completed_seeds]
        print(f"Total tasks: {len(test_tasks)}. Remaining tasks to process: {len(tasks_to_process)}.")
    else:
        tasks_to_process = test_tasks[args.start_id:]

    if INITIAL_INFO.get("debug"):
        tasks_to_process = tasks_to_process[:5]
        print(f"Debug mode enabled, processing only {len(tasks_to_process)} tasks.")

    # --- 3. Initialize Strategy and VLM Model ---
    try:
        strategy = get_prompt_strategy(INITIAL_INFO["format"])
    except ValueError as e:
        print(f"Error: {e}. Aborting.")
        return

    if INITIAL_INFO["model"].startswith("Qwen"):
        qwen_pipe = initialize_qwen_vlm(INITIAL_INFO["model"])

    # --- 4. Main Loop ---
    summary_file_path = os.path.join(main_results_dir, "summary_results.json")
    if not tasks_to_process:
        print("\nAll tasks are already completed. No further action needed.")
        return

    print(f"\n{'='*50}\nStarting to process {len(tasks_to_process)} tasks...\nResults will be saved in: {main_results_dir}\n{'='*50}\n")

    for seed in tqdm(tasks_to_process, desc="Processing Tasks"):
        img_dir_for_task = os.path.join(main_results_dir, f"imgs_{seed.replace(':', '-')}")
        task_result = process_single_task_with_retries(seed, simulator, img_dir_for_task, INITIAL_INFO, strategy)
        all_results.append(task_result)

        try:
            with open(summary_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error: Could not save JSON summary file: {e}")

        # MODIFICATION: The image folder is no longer deleted after each task.
        # This preserves the generated images and logs for inspection.
        # if os.path.exists(img_dir_for_task):
        #     try:
        #         shutil.rmtree(img_dir_for_task)
        #     except Exception as e:
        #         print(f"Warning: Could not delete temporary folder {img_dir_for_task}: {e}")

    print(f"\n{'='*50}\nAll tasks processed!\nFinal summary results have been successfully saved to: {summary_file_path}\n{'='*50}")

if __name__ == "__main__":
    main()