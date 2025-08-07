import os
import re
import time
import datetime
import random
import json
import imageio
import argparse
import shutil # Added for directory cleanup consistent with phyre example
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pooltool as pt

from deepphy.config import Config
from deepphy.provider.llm.base import *
from deepphy.utils.img_utils import *
from deepphy.utils.dir_utils import *
from deepphy.planner.prompt_strategy_04_pooltool import *

CONFIG_JSON_PATH='conf/env_config_04_pooltool.json'
config = Config()
config.load_env_config(CONFIG_JSON_PATH)

# Removed global constants here as they are now managed by the PromptStrategy classes

def get_cue_ball_reposition_point(table: pt.Table) -> np.ndarray:
    """Calculates the repositioning point for the cue ball (typically behind the head string)."""
    x_reposition = table.w / 2
    y_reposition = table.l / 4
    z_ball = pt.BallParams().R # Use default ball radius for Z-coordinate
    return np.array([x_reposition, y_reposition, z_ball])

def check_for_potted_balls(system: pt.System) -> dict:
    """
    Checks if any balls were potted based on the system's events.
    Returns a dictionary with ball_id as key and True/False as value.
    """
    potted_status = {}

    for ball_id, ball in system.balls.items():

        pocket_events = pt.events.filter_events(
            system.events,
            pt.events.by_type(pt.EventType.BALL_POCKET),
            pt.events.by_ball(ball_id),
        )
        potted_status[ball_id] = bool(len(pocket_events))

    return potted_status

def setup_pooltool_game() -> pt.System:
    """Initializes a new 9-ball game system."""
    initial_table = pt.Table.default()
    initial_balls_dict = pt.get_rack(pt.GameType.NINEBALL, initial_table)
    initial_cue = pt.Cue(cue_ball_id="cue")

    # Use OrderedDict to ensure consistent iteration order for balls.
    game_system = pt.System(table=initial_table, balls=OrderedDict(initial_balls_dict), cue=initial_cue)
    return game_system

# --- Mock VLM Response Function ---
def get_mock_vlm_response(attempt_history: list, config: Config) -> str:
    """
    Generates a mock VLM response for testing purposes.
    It attempts to return a previously untried speed/strikespot combination.
    """
    tried_combinations = set()
    for a in attempt_history:
        # Note: 'chosen_speed_name' and 'chosen_strikespot_name' are added to attempt_history
        # This mock function still uses attempt_history to ensure it tries new combinations,
        # even if the VLM itself no longer explicitly sees this history in the prompt.
        if a.get('chosen_speed_name') and a.get('chosen_strikespot_name'):
            tried_combinations.add((a['chosen_speed_name'], a['chosen_strikespot_name']))

    available_speeds = list(config.speed_mapping.keys())
    available_strikespots = list(config.strikespot_mapping.keys())

    chosen_speed_name_mock = None
    chosen_strikespot_name_mock = None

    # Try to find a new, untried combination
    # Limit attempts to avoid infinite loop if all combinations are tried
    for _ in range(len(available_speeds) * len(available_strikespots) * 2):
        temp_speed = random.choice(available_speeds)
        temp_strikespot = random.choice(available_strikespots)
        if (temp_speed, temp_strikespot) not in tried_combinations:
            chosen_speed_name_mock = temp_speed
            chosen_strikespot_name_mock = temp_strikespot
            break

    if chosen_speed_name_mock is None: # If all combinations are tried or no new one found, allow repetition
        chosen_speed_name_mock = random.choice(available_speeds)
        chosen_strikespot_name_mock = random.choice(available_strikespots)

    vlm_response = f"Speed: {chosen_speed_name_mock}, Strikespot: {chosen_strikespot_name_mock}."
    print(f"  (Mock VLM) Generated action: '{vlm_response}'")
    time.sleep(0.5) # Simulate processing time
    return vlm_response

def process_single_game_with_retries(game_id: int, initial_game_system: pt.System, game_results_dir: str, INITIAL_INFO: dict, strategy: PromptStrategy) -> dict:
    """
    Processes a single billiards game with retry logic.
    Provides cumulative history (images and text feedback) to the VLM on each new attempt.
    """

    # 1. Game Task Initialization
    os.makedirs(game_results_dir, exist_ok=True)

    # Deep copy the initial system to ensure each game starts from a fresh state
    current_game_system = initial_game_system.copy()

    # 2. Generate Initial Table Image (no grid)
    # The initial image is still generated and stored locally, but will no longer be passed to the VLM.
    # It's kept for records/debugging if needed outside the VLM interaction.
    initial_image_path = os.path.join(game_results_dir, "game_initial_state.png")
    plot_static_pool_table(current_game_system, filename=initial_image_path) # No grid
    if not os.path.exists(initial_image_path):
        print(f"Game {game_id} aborted: Failed to create initial image.")
        return { "game_id": game_id, "is_solved": False, "notes": "Failed to create initial image." }

    # 3. Attempt Loop
    is_game_solved = False
    attempt_history = [] # Stores complete history of all attempts

    for attempt_num in range(1, config.max_attempts_per_game + 1):
        print(f"\n--- Game {game_id} Attempt {attempt_num}/{config.max_attempts_per_game} (Format: {INITIAL_INFO.get('format')}) ---")

        cue_ball_id = current_game_system.cue.cue_ball_id

        # Handle cue ball potting logic (based on current_game_system state AFTER previous shot)
        if cue_ball_id not in current_game_system.balls:
            print(f"[{attempt_num}] Cue ball '{cue_ball_id}' was potted in the previous turn. Repositioning...")
            reposition_point = get_cue_ball_reposition_point(current_game_system.table)

            # Create a new cue ball object and add it back to the system's balls dictionary
            new_cue_ball = pt.Ball.create(cue_ball_id, xy=reposition_point[:2])
            current_game_system.balls[cue_ball_id] = new_cue_ball
            print(f"  Cue ball re-added and repositioned to {reposition_point[:2]}.")
        else:
            print(f"[{attempt_num}] Cue ball '{cue_ball_id}' was not potted in the previous turn. Retaining its position.")

        # Stop all remaining balls' motion and reset history for this simulation run
        current_game_system.stop_balls()
        current_game_system.reset_history() # Clear events for this run

        # Get current table image for VLM input (no grid)
        current_table_image_path = os.path.join(game_results_dir, f"attempt_{attempt_num}_current_table.png")
        plot_static_pool_table(current_game_system, filename=current_table_image_path) # No grid
        if not os.path.exists(current_table_image_path):
            print(f"Game {game_id} Attempt {attempt_num} aborted: Failed to create current table image.")
            break

        # --- MODIFICATION START ---
        # Prepare VLM Input: ONLY the current table image is passed.
        image_descriptions = [
            {'path': current_table_image_path, 'label': "Image (Current Table State): Analyze this image to determine your next move."},
        ]
        # --- MODIFICATION END ---

        # Generate prompts using the selected strategy
        # Note: `attempt_history` is still passed to the strategy, but the strategy's `generate_prompts`
        # method has been modified NOT to include this history in the VLM's prompt text.
        # It's kept here for potential future use or for mock VLM to learn from (as seen in `get_mock_vlm_response`).
        system_prompt, user_prompt = strategy.generate_prompts(attempt_history, image_descriptions)

        print(f"Preparing to call VLM for attempt {attempt_num}.")
        model_name = INITIAL_INFO.get("model")
        vlm_response = ""

        if model_name.lower() == "mock":
            vlm_response = get_mock_vlm_response(attempt_history, config)
        else:
            vlm_response = get_model_response(image_descriptions=image_descriptions, system_prompt=system_prompt, user_prompt=user_prompt, model_name=model_name, qwen_pipe=qwen_pipe)

        print("VLM raw response: ", str(vlm_response))

        # PARSE VLM RESPONSE - DELEGATE TO STRATEGY
        parsed_output = strategy.parse_response(vlm_response)


        parsed_action_params = parsed_output.get("action")
        chosen_speed_name = parsed_output.get("parsed_speed_name")
        chosen_strikespot_name = parsed_output.get("parsed_strikespot_name")

        simulation_outcome = "UNKNOWN_ERROR" # Default state
        potted_balls_in_this_run = [] # Initialize as empty list

        # Only proceed with simulation if action parameters were successfully parsed
        if parsed_action_params and chosen_speed_name and chosen_strikespot_name:
            V0, a, b = parsed_action_params
            print(f"  (Attempt {attempt_num}) Simulating action: Speed={chosen_speed_name} (V0={V0}), Strikespot={chosen_strikespot_name} (a={a}, b={b})...")

            # Find the lowest-numbered ball on the table to target
            available_object_balls = []
            for ball_id in current_game_system.balls.keys():
                if ball_id != cue_ball_id: # Exclude cue ball
                    try:
                        num_id = int(ball_id)
                        if 1 <= num_id <= 15: # Standard numbered ball range
                            available_object_balls.append(ball_id)
                    except ValueError:
                        pass # Ignore non-numeric ball IDs

            target_ball_id = None
            if available_object_balls:
                available_object_balls.sort(key=lambda x: int(x)) # Sort numerically
                target_ball_id = available_object_balls[0] # Select the lowest-numbered ball as target
                print(f"  Next target ball (lowest ID on table): '{target_ball_id}'.")
            else:
                print("  No numbered balls left on the table to target.")
                simulation_outcome = "NO_TARGET_BALLS_LEFT"
                is_game_solved = ("9" not in current_game_system.balls.keys()) # Game ends, check if 9-ball was already potted
                break # End current game attempt loop

            # Set cue parameters and aim at the target ball
            current_game_system.cue.set_state(
                V0=V0,
                phi=pt.aim.at_ball(current_game_system, target_ball_id),
                a=a,
                b=b
            )

            # Execute simulation
            try:
                pt.simulate(current_game_system, inplace=True)

                # Check simulation results
                current_run_potted_status = check_for_potted_balls(current_game_system)

                for ball_id, is_potted in current_run_potted_status.items():
                    if is_potted:
                        potted_balls_in_this_run.append(ball_id)

                if "9" in potted_balls_in_this_run:
                    simulation_outcome = "9_BALL_POTTED"
                    is_game_solved = True
                elif cue_ball_id in potted_balls_in_this_run:
                    simulation_outcome = "CUE_BALL_POTTED"
                elif not potted_balls_in_this_run:
                    simulation_outcome = "NO_BALLS_POTTED"
                else:
                    simulation_outcome = "OTHER_BALLS_POTTED" # Other balls potted, but not 9-ball or cue ball

                # Update current_game_system.balls to logically remove potted balls
                new_balls_on_table_dict = OrderedDict()
                for ball_id, ball_obj in current_game_system.balls.items():
                    if ball_id not in potted_balls_in_this_run:
                        new_balls_on_table_dict[ball_id] = ball_obj
                    else:
                        print(f"  Removing ball '{ball_id}' from system as it was potted this turn.")
                current_game_system.balls = new_balls_on_table_dict # Update system's ball dictionary

            except ValueError as e:
                # Catch the specific ValueError from Numba/pooltool
                if "cannot assign slice of shape (2,) from input of shape (4,)" in str(e):
                    print(f"  [Error] Simulation failed due to Numba/Pooltool root solver issue: {e}")
                    simulation_outcome = "SIMULATION_CRASHED_ROOT_SOLVER_ISSUE"
                else:
                    # Re-raise if it's a different ValueError, or handle more generically
                    print(f"  [Error] Simulation failed with an unexpected ValueError: {e}")
                    simulation_outcome = "UNEXPECTED_SIMULATION_ERROR"

                # Ensure no balls are marked as potted if simulation failed
                potted_balls_in_this_run = []
                is_game_solved = False # Game is not solved if simulation crashed
                # current_game_system.balls will remain unchanged as the operations to update it were inside the try block.

            except Exception as e:
                # Catch any other unexpected errors during simulation
                print(f"  [Error] Simulation failed with an unexpected error: {e}")
                simulation_outcome = "UNEXPECTED_SIMULATION_ERROR"
                potted_balls_in_this_run = []
                is_game_solved = False # Game is not solved if simulation crashed
                # current_game_system.balls will remain unchanged.

            print(f"  (Attempt {attempt_num}) Simulation result: {simulation_outcome}. Balls potted this turn: {potted_balls_in_this_run}")

            # Print detailed status of remaining balls on table
            print("\n  Balls currently on table (after potting and removal):")
            if not current_game_system.balls:
                print("    No balls remaining on table.")
            else:
                sorted_ball_ids = sorted(current_game_system.balls.keys(), key=lambda x: (x.isalpha(), int(x) if x.isdigit() else float('inf')))
                for ball_id in sorted_ball_ids:
                    ball_obj = current_game_system.balls[ball_id]
                    print(f"    Ball '{ball_id}': Position={ball_obj.xyz[:2]}")

            # Optional: Visualize current simulation (only for debugging)
            # print(f"[{attempt_num}] Opening GUI. Close to continue...")
            # pt.show(current_game_system)
            # print(f"[{attempt_num}] GUI closed.")

        else: # If parsed_action_params is None or other components are missing
            simulation_outcome = "INVALID_VLM_ACTION"
            print(f"  (Attempt {attempt_num}) VLM failed to provide a valid action. Result: {simulation_outcome}\n")

        # Record current attempt in history - DELEGATE TO STRATEGY
        base_attempt_data = {
            "attempt_number": attempt_num,
            "vlm_response": vlm_response, # Raw VLM response (can contain prediction + action)
            "chosen_speed_name": chosen_speed_name,
            "chosen_strikespot_name": chosen_strikespot_name,
            "parsed_action_params": parsed_action_params,
            "simulation_outcome": simulation_outcome,
            "potted_balls_in_this_run": potted_balls_in_this_run,
            "is_game_solved_in_this_attempt": is_game_solved,
        }
        # Strategy updates base_attempt_data with its specific fields (e.g., 'prediction' for WM)
        current_attempt_data = strategy.update_attempt_data(base_attempt_data, parsed_output)
        attempt_history.append(current_attempt_data)

        # Check if game is solved or should be aborted
        if is_game_solved:
            print(f"*** Game {game_id} SOLVED, 9-ball potted in {attempt_num} attempts! ***\n")
            break
        elif simulation_outcome == "NO_TARGET_BALLS_LEFT" and not is_game_solved:
            print(f"--- Game {game_id} aborted: No numbered balls left on table and 9-ball was not potted. ---")
            break
        elif simulation_outcome == "INVALID_VLM_ACTION":
            # If invalid action, and it's the last attempt, then abort
            if attempt_num == config.max_attempts_per_game:
                print(f"--- Game {game_id} aborted: VLM continuously output invalid actions. ---")
            # else: continue to next attempt if not last
        elif simulation_outcome == "SIMULATION_CRASHED_ROOT_SOLVER_ISSUE":
            # If simulation crashed due to the root solver issue, treat it like an invalid action
            # and allow retries, but if it's the last attempt, abort.
            if attempt_num == config.max_attempts_per_game:
                print(f"--- Game {game_id} aborted: Simulation repeatedly crashed due to root solver issue. ---")
        elif attempt_num == config.max_attempts_per_game:
            print(f"--- Game {game_id} aborted: Reached maximum {config.max_attempts_per_game} attempts without potting 9-ball. ---")

    # Return final result
    final_attempt_count = attempt_num
    final_result = {
        "game_id": game_id,
        "is_solved": is_game_solved,
        "total_attempts": final_attempt_count,
        "output_dir": game_results_dir,
        "attempt_history": attempt_history,
        "notes": f"Game solved in {final_attempt_count} attempts." if is_game_solved else f"Game not solved after {final_attempt_count} attempts."
    }

    return final_result



def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM agent strategy in billiards games.")
    parser.add_argument(
        "--model",
        type=str,
        default="mock", # Changed default to "mock" for easier local testing
        help="The VLM model to use for solving the puzzle."
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=100, # Default to 3 independent game runs
        help="Number of independent game sessions to simulate."
    )
    parser.add_argument(
        "--format",
        type=str,
        default="VLA",
        choices=["VLA", "WM"],
        help="The prompt format to use: VLA (Vision-Language-Action) or WM (World Model)."
    )
    parser.add_argument(
        "--debug",
        action="store_true", # Boolean flag
        help="Enable debug mode (processes only a few tasks)."
    )
    parser.add_argument(
        "--reuse_log_dir",
        type=str,
        default=None,
        help="Specify an existing log directory to resume a run, e.g., /tmp_log/pooltool_2025-06-09_14-25-39"
    )
    parser.add_argument(
        "--LOG_BASE_PATH",
        type=str,
        default="tmp_log/pooltool/",
        help="Base directory for log files."
    )

    args = parser.parse_args()

    # global qwen_pipe is declared outside main, but assigned inside.
    # It should be initialized only if Qwen model is chosen.
    global qwen_pipe
    qwen_pipe = None

    INITIAL_INFO_from_args = {k: v for k, v in vars(args).items()}
    print(f"Running with the following configuration: {INITIAL_INFO_from_args}")

    LOG_BASE_PATH = INITIAL_INFO_from_args["LOG_BASE_PATH"]
    all_results = []
    games_to_process = []

    if args.reuse_log_dir:
        # --- Resume Run Mode ---
        main_results_dir = args.reuse_log_dir
        print(f"\nDetected --reuse_log_dir. Attempting to resume run from '{main_results_dir}'.")
        os.makedirs(main_results_dir, exist_ok=True)
        summary_file_path = os.path.join(main_results_dir, "summary_results.json")

        completed_game_ids = set()

        if os.path.exists(summary_file_path):
            print(f"Loading existing results from '{summary_file_path}'...")
            try:
                with open(summary_file_path, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)

                if not all_results: raise ValueError("Log file is empty.")

                # Load initial configuration from log file
                INITIAL_INFO = all_results[0]
                print(f"Successfully loaded log file. Using configuration from file: {INITIAL_INFO}")

                # Check if configurations match and issue warnings (or force override as needed)
                # Forcing usage of log file config to maintain consistency
                for key in ["model", "num_games", "format"]: # Include 'format' in synchronization
                    if INITIAL_INFO.get(key) != getattr(args, key):
                        print(f"Warning: Command line {key} ('{getattr(args, key)}') does not match log file ('{INITIAL_INFO.get(key)}').")
                        setattr(args, key, INITIAL_INFO.get(key))
                        print(f"--> Forced usage of log file setting: {key} = '{getattr(args, key)}'")

                for result in all_results[1:]: # First element is INITIAL_INFO
                    if "game_id" in result:
                        completed_game_ids.add(result["game_id"])
                print(f"Found {len(completed_game_ids)} already processed games.")

            except (json.JSONDecodeError, IndexError, ValueError) as e:
                print(f"Error: Could not parse '{summary_file_path}' ({e}). Starting as a new run in the specified directory.")
                INITIAL_INFO = INITIAL_INFO_from_args # Reset to args
                all_results = [INITIAL_INFO] # Reset
        else:
            print(f"Warning: summary_results.json not found in '{main_results_dir}'. Starting as a new run in the specified directory.")
            INITIAL_INFO = INITIAL_INFO_from_args # Reset to args
            all_results = [INITIAL_INFO] # Reset

        # Generate all game IDs to process
        all_game_ids = list(range(INITIAL_INFO["num_games"])) # Use INITIAL_INFO for num_games
        games_to_process = [gid for gid in all_game_ids if gid not in completed_game_ids]
        print(f"Total games: {len(all_game_ids)}. Remaining games to process: {len(games_to_process)}.")

    else:
        # --- New Run Mode ---
        INITIAL_INFO = INITIAL_INFO_from_args # Use args config for new run
        model = INITIAL_INFO.get("model")
        game_format = INITIAL_INFO.get("format") # Get format from INITIAL_INFO
        # Include format in the log path for new runs
        full_log_path = os.path.join(LOG_BASE_PATH, model, game_format)
        main_results_dir = create_main_results_dir(log_dir=full_log_path) # Uses timestamp internally
        print(f"Main results directory created: {main_results_dir}")
        all_results = [INITIAL_INFO] # First element is config info

        games_to_process = list(range(args.num_games)) # From 0 to num_games-1

    if INITIAL_INFO.get("debug"):
        games_to_process = games_to_process[:1] # Debug mode processes only 1 game
        print(f"Debug mode enabled, processing {len(games_to_process)} game(s).")

    # Initialize Prompt Strategy based on the chosen format
    try:
        strategy = get_prompt_strategy(INITIAL_INFO["format"])
    except ValueError as e:
        print(f"Error: {e}. Aborting.")
        return

    # Initialize VLM model (if not mock)
    if INITIAL_INFO["model"].startswith("Qwen"):
        qwen_pipe = initialize_qwen_vlm(INITIAL_INFO["model"])


    print("\n" + "="*50)
    print(f"Starting to process {len(games_to_process)} game(s)...")
    print(f"Results will be saved in: {main_results_dir}")
    print("="*50 + "\n")

    summary_file_path = os.path.join(main_results_dir, "summary_results.json")

    # Pre-set an initial game system template, deep copy for each game
    initial_game_system_template = setup_pooltool_game()

    if not games_to_process:
        print("\nAll games already completed, no further action required.")
        return

    for game_id in tqdm(games_to_process, desc="Processing Games"):
        print(f"\n--- Starting to process game: {game_id} ---")

        # Create a temporary image directory for the current game, which will be cleaned up later
        game_img_dir = os.path.join(main_results_dir, f"imgs_game_{game_id}")
        os.makedirs(game_img_dir, exist_ok=True)

        # Pass the strategy object to the processing function
        game_result = process_single_game_with_retries(game_id, initial_game_system_template, game_img_dir, INITIAL_INFO, strategy)
        all_results.append(game_result)

        try:
            with open(summary_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error: Could not save JSON summary file: {e}")

        # Clean up temporary image directory after each game, consistent with phyre example
        if os.path.exists(game_img_dir):
            try:
                shutil.rmtree(game_img_dir)
            except Exception as e:
                print(f"Warning: Could not delete temporary folder {game_img_dir}: {e}")

    print("\n" + "="*50)
    print("All games processed!")
    print(f"Final summary results successfully saved to: {summary_file_path}")
    print("="*50)

if __name__ == "__main__":
    main()
