# deepphy/runner/03_kinetix_runner.py

import argparse
import base64
import json
import os
import time

# Third-party imports
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from openai import OpenAI

# Local application/library specific imports
from deepphy.config import Config
from deepphy.provider.llm.base import get_model_response, initialize_qwen_vlm
from deepphy.planner.prompt_strategy_03_kinetix import generate_prompts_and_images

from suite.kinetix.environment import make_kinetix_env
from suite.kinetix.environment.utils import ActionType, ObservationType
from suite.kinetix.render import make_render_pixels_unanno, make_render_pixels_anno
from suite.kinetix.util import load_from_json_file

# --- Configuration Loading ---
CONFIG_JSON_PATH = 'conf/env_config_03_kinetix.json'
config = Config()
config.load_env_config(CONFIG_JSON_PATH)

# ==============================================================================
# SECTION 1: Helper Functions
# ==============================================================================

def add_item_to_limited_list(target_list, item, max_size):
    """Adds an item to a list, maintaining a maximum size by removing the oldest item."""
    target_list.append(item)
    if len(target_list) > max_size:
        target_list.pop(0)
    return target_list

def load_active_entities_from_json(json_path):
    """Reads a level JSON to determine the number of active joints and thrusters."""
    with open(json_path, "r") as f:
        data = json.load(f)
    env_state = data["env_state"]
    num_active_joints = sum(joint.get("motor_on", False) for joint in env_state["joint"])
    num_active_thrusters = sum(thruster.get("active", False) for thruster in env_state["thruster"])
    return num_active_joints, num_active_thrusters

def _parse_action_response(action_text):
    """
    Robustly parses the LLM's string response to extract an action list.
    Handles JSON format, plain comma-separated values, and other variations.
    """
    try:
        start_index = action_text.rfind('[')
        end_index = action_text.rfind(']')
        if start_index != -1 and end_index != -1 and start_index < end_index:
            json_str = action_text[start_index : end_index + 1]
            return [int(x) for x in json.loads(json_str)]
    except (json.JSONDecodeError, TypeError, ValueError):
        cleaned_text = action_text.strip().strip("[]` ")
        if cleaned_text:
            try:
                return [int(x.strip()) for x in cleaned_text.split(",") if x.strip()]
            except ValueError:
                print(f"  LLM API: Fallback parsing failed. Could not convert response to integers: '{action_text}'")

    print(f"  LLM API: Could not parse a valid action list from response: '{action_text}'")
    return None

# ==============================================================================
# SECTION 2: Core Evaluation Logic
# ==============================================================================

def get_llm_action(model_name, num_active_joints, num_active_thrusters,
                   previous_actions, distances, image_history,
                   current_step, current_image_paths, last_action,
                   prompt_format, is_annotated, qwen_pipe=None):
    """
    Gets an action from the specified LLM, handling different model and prompt types.
    """
    raw_response = "N/A"
    action_list = None

    if model_name == "mock":
        print("  Mock Mode: Generating random action.")
        raw_response = "mock_action"
        mock_rng = jax.random.PRNGKey(int(time.time() * 1000))
        key_joint, key_thruster = jax.random.split(mock_rng)
        joint_actions = jax.random.randint(key_joint, shape=(num_active_joints,), minval=0, maxval=3).tolist()
        thruster_actions = jax.random.randint(key_thruster, shape=(num_active_thrusters,), minval=0, maxval=2).tolist()
        action_list = joint_actions + thruster_actions

    elif config.evaluation_settings['debug_mode']:
        raw_response = "debug_mode_zero_action"
        action_list = [0] * (num_active_joints + num_active_thrusters)

    else:
        system_prompt, user_prompt, image_descriptions = generate_prompts_and_images(
            num_active_joints, num_active_thrusters, previous_actions,
            distances, image_history, current_step, current_image_paths, last_action,
            prompt_format, is_annotated
        )
        action_text = get_model_response(
            image_descriptions, system_prompt, user_prompt, model_name, qwen_pipe=qwen_pipe
        )
        raw_response = action_text or "empty_response"

        if not action_text:
            print("  LLM API: Received empty response. Will use default zero action.")
        else:
            print(f"  LLM API: Raw response received: {action_text}")
            action_list = _parse_action_response(action_text)

    # --- Action Post-processing ---
    if action_list is None:
        print("  LLM API: Could not determine a valid action. Using default zero action.")
        action_list = [0] * (num_active_joints + num_active_thrusters)

    joint_actions_final = action_list[:num_active_joints]
    thruster_actions_final = action_list[num_active_joints:]

    joint_action_list = list(joint_actions_final) + [0] * (4 - num_active_joints)
    thruster_action_list = list(thruster_actions_final) + [0] * (2 - num_active_thrusters)
    final_action_list = joint_action_list + thruster_action_list
    action_array = jnp.array(final_action_list, dtype=jnp.int32)

    print(f"  LLM API: Final parsed action array: {action_array}")
    return action_array, raw_response


def render_and_save_frame(renderer, env_state, step_index, base_path, is_annotated):
    """Renders the current state, saves image(s), and returns their paths."""
    image_paths = {}
    if is_annotated:
        pixels, joint_coords, joint_is_motor, thruster_coords, thruster_is_active = renderer(env_state)
        img_to_plot = np.array(pixels.astype(jnp.uint8).transpose(1, 0, 2)[::-1])

        # Save unannotated image
        plt.figure(figsize=(4, 4), dpi=150)
        plt.imshow(img_to_plot)
        plt.axis('off')
        unannotated_path = os.path.join(base_path, f"frame_{step_index:04d}_unannotated.png")
        plt.savefig(unannotated_path, bbox_inches='tight', pad_inches=0)
        image_paths['unannotated'] = unannotated_path

        # Add labels and save annotated image
        motor_idx = 1
        for i in range(len(joint_coords)):
            if joint_is_motor[i]:
                jax_x, jax_y = joint_coords[i]
                plot_x, plot_y = jax_x, img_to_plot.shape[0] - jax_y
                plt.text(plot_x, plot_y, f'M{motor_idx}', color='white', fontsize=8, fontweight='bold',
                         ha='center', va='center', bbox=dict(facecolor='black', alpha=0.6, pad=0.5, boxstyle='round,pad=0.2'))
                motor_idx += 1

        thruster_idx = 1
        for i in range(len(thruster_coords)):
            if thruster_is_active[i]:
                jax_x, jax_y = thruster_coords[i]
                plot_x, plot_y = jax_x, img_to_plot.shape[0] - jax_y
                plt.text(plot_x, plot_y, f'T{thruster_idx}', color='white', fontsize=8, fontweight='bold',
                         ha='center', va='center', bbox=dict(facecolor='black', alpha=0.6, pad=0.5, boxstyle='round,pad=0.2'))
                thruster_idx += 1

        annotated_path = os.path.join(base_path, f"frame_{step_index:04d}_annotated.png")
        plt.savefig(annotated_path, bbox_inches='tight', pad_inches=0)
        image_paths['annotated'] = annotated_path
        plt.close()
        print(f"  Saved unannotated and annotated frames for step {step_index}")
    else: # Unannotated
        pixels_numpy = renderer(env_state)
        plt.imshow(pixels_numpy.astype(jnp.uint8).transpose(1, 0, 2)[::-1])
        unannotated_path = os.path.join(base_path, f"frame_{step_index:04d}.png")
        plt.savefig(unannotated_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        image_paths['unannotated'] = unannotated_path
        print(f"  Saved frame to {unannotated_path}")

    return image_paths


def level_eval(json_path, log_dir, model_name, repeat_count,
               prompt_format, is_annotated, qwen_pipe=None):
    """
    Runs evaluation for a single level, logging detailed step-by-step information.
    """
    print(f"\n--- Evaluating level: {os.path.basename(json_path)} ---")
    json_path_short = json_path.replace("suite/kinetix/levels/", "")
    level, static_env_params, env_params = load_from_json_file(json_path_short)
    num_active_joints, num_active_thrusters = load_active_entities_from_json(json_path)

    print(f"  Level Config: Active Joints={num_active_joints}, Active Thrusters={num_active_thrusters}, Timesteps={env_params.max_timesteps}")
    print(f"  Run Config: Prompt Format='{prompt_format}', Annotated={is_annotated}")

    env = make_kinetix_env(
        action_type=ActionType.MULTI_DISCRETE,
        observation_type=ObservationType.PIXELS,
        reset_fn=lambda rng: level,
        env_params=env_params,
        static_env_params=static_env_params
    )
    renderer = make_render_pixels_anno(env_params, static_env_params) if is_annotated else make_render_pixels_unanno(env_params, static_env_params)
    print("  Kinetix environment and renderer created.")

    master_rng, rng_reset = jax.random.split(jax.random.PRNGKey(0))
    obs, env_state = env.reset(rng_reset, env_params)

    done, reward, info = False, 0.0, {}
    last_action = jnp.zeros(6, dtype=jnp.int32)
    action_history, distance_history, image_history = [], [], []
    history_size = config.evaluation_settings['history_length']
    step_details_log = []

    # Save initial frame(s)
    current_image_paths = render_and_save_frame(renderer, env_state, 0, log_dir, is_annotated)

    max_steps = (env_params.max_timesteps + 1) // repeat_count
    for i in range(1, max_steps + 1):
        print("\n" + "---"*4 + f"--- Step {i}/{max_steps} ---" + "---"*4)
        if done:
            print(f"  Environment marked as done. Ending evaluation for this level.")
            break

        action, llm_raw_response = get_llm_action(
            model_name, num_active_joints, num_active_thrusters,
            action_history, distance_history, image_history,
            current_step=i, current_image_paths=current_image_paths,
            last_action=last_action, prompt_format=prompt_format,
            is_annotated=is_annotated, qwen_pipe=qwen_pipe
        )

        _, rng_loop = jax.random.split(master_rng)
        for j in range(repeat_count):
            rng_loop, step_rng = jax.random.split(rng_loop)
            obs, env_state, reward, done, info = env.step(step_rng, env_state, action, env_params)
            if done:
                print(f"    Sub-step {j+1}/{repeat_count}: Environment done.")
                break

        distance_value = round(info.get('distance', jnp.array(0.0)).item(), 2)
        print(f"  Step Result: Reward={reward}, Done={done}, Distance={distance_value}")

        step_details_log.append({
            "step": i,
            "llm_raw_response": llm_raw_response,
            "parsed_action": action.tolist(),
            "environment_feedback": {"reward": float(reward), "done": bool(done), "distance": distance_value, "info": str(info)}
        })

        action_history = add_item_to_limited_list(action_history, action, history_size)
        distance_history = add_item_to_limited_list(distance_history, distance_value, history_size)
        history_img_path = current_image_paths.get('annotated', current_image_paths['unannotated'])
        image_history = add_item_to_limited_list(image_history, history_img_path, history_size)
        last_action = action

        if not done:
            current_image_paths = render_and_save_frame(renderer, env_state, i, log_dir, is_annotated)

    level_name = os.path.basename(json_path).replace(".json", "")
    end_info = {
        "level": level_name,
        "final_reward": float(reward),
        "final_done": bool(done),
        "final_info": str(info),
        "steps": step_details_log
    }
    print(f"--- Evaluation for level '{level_name}' complete. ---")
    return end_info

def run_evaluation_suite(levels_folder, log_dir, model_name, repeat_count,
                         prompt_format, is_annotated, qwen_pipe, skip_levels):
    """
    Manages the evaluation for a whole folder of levels.
    """
    all_level_files = sorted([os.path.join(root, file) for root, _, files in os.walk(levels_folder) for file in files if file.endswith(".json")])
    levels_to_process = [f for f in all_level_files if os.path.basename(f).replace(".json", "") not in skip_levels]

    if not levels_to_process:
        print("\nAll levels have already been evaluated. Nothing to do.")
        return

    print(f"\nFound {len(all_level_files)} total levels. Will process {len(levels_to_process)} remaining levels.")
    end_info_path = os.path.join(log_dir, "end_info.json")

    for i, json_path in enumerate(levels_to_process, 1):
        level_name = os.path.basename(json_path).replace(".json", "")
        print(f"\n{'='*20} Processing Level {i}/{len(levels_to_process)}: {level_name} {'='*20}")

        level_log_dir = os.path.join(log_dir, level_name, "")
        os.makedirs(level_log_dir, exist_ok=True)
        print(f"  Log path for this level: {level_log_dir}")

        end_info = level_eval(json_path, level_log_dir, model_name, repeat_count,
                              prompt_format, is_annotated, qwen_pipe)

        with open(end_info_path, "r+") as f:
            data = json.load(f)
            data["results"].append(end_info)
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()

        print(f"Level '{level_name}' results saved to {end_info_path}")

    print(f"\n--- Evaluation suite complete. All results are in {end_info_path} ---")

# ==============================================================================
# SECTION 3: Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Kinetix evaluation with flexible settings.")
    # Arguments for a new run
    parser.add_argument(
        "level_type", type=str, nargs='?', help="Level type ('s', 'm', 'l'). Required for a new run."
    )
    parser.add_argument(
        '--model', type=str, default='mock',
        help=f"Model name. Default: MOCK"
    )
    parser.add_argument(
        '--format', type=str, default='VLA', choices=['VLA', 'WM'],
        help="Prompt format ('VLA' or 'WM'). Default: VLA"
    )
    parser.add_argument(
        '--annotated', type=str, default='unanno', choices=['anno', 'unanno'],
        help="Annotation mode ('anno' for annotated, 'unanno' for unannotated). Default: unanno"
    )
    # Argument for resuming a run
    parser.add_argument(
        '--reuse_log_dir', type=str, default=None,
        help="Path to a previous log directory to resume an incomplete run."
    )
    args = parser.parse_args()

    completed_levels = set()

    if args.reuse_log_dir:
        print(f"--- Resuming evaluation from log directory: {args.reuse_log_dir} ---")
        log_dir = args.reuse_log_dir
        end_info_path = os.path.join(log_dir, "end_info.json")

        if not os.path.exists(end_info_path):
            print(f"Error: 'end_info.json' not found in '{log_dir}'. Cannot resume.")
            exit(1)

        with open(end_info_path, 'r') as f:
            metadata = json.load(f).get('metadata', {})

        model_name = metadata.get('model')
        repeat_count = metadata.get('repeat_actions')
        levels_folder = metadata.get('levels_folder')
        prompt_format = metadata.get('prompt_format', 'VLA')
        annotation_mode = metadata.get('annotation_mode', 'unanno')
        is_annotated = (annotation_mode == 'anno')

        if not all([model_name, repeat_count, levels_folder]):
            print("Error: 'end_info.json' is missing required metadata (model, repeat_actions, levels_folder).")
            exit(1)

        with open(end_info_path, 'r') as f:
            completed_levels = {r['level'] for r in json.load(f).get('results', [])}

        print("--- Settings Loaded from Metadata ---")
        print(f"  Model: '{model_name}', Levels Folder: {levels_folder}")
        print(f"  Repeat Count: {repeat_count}, Prompt Format: {prompt_format}, Annotated: {is_annotated}")
        print(f"  Found {len(completed_levels)} completed level(s) to skip.")

    else:
        if not args.level_type:
            parser.error("The 'level_type' argument is required for a new run.")

        print("--- Starting a new evaluation run ---")
        model_name = args.model
        level_type = args.level_type
        prompt_format = args.format
        annotation_mode = args.annotated
        is_annotated = (annotation_mode == 'anno')

        repeat_count = config.evaluation_settings['repeat_actions']
        levels_folder = os.path.join(config.paths['levels_base'], level_type)
        now_time = time.strftime("%Y%m%d_%H%M%S")
        log_subfolder = f"{prompt_format}/{annotation_mode}"
        log_dir = os.path.join(config.paths['log_base'], model_name, log_subfolder, f"levels_{level_type}", now_time, "")

        os.makedirs(log_dir, exist_ok=True)

        print(f"  Model: '{model_name}', Level Type: '{level_type}'")
        print(f"  Prompt Format: '{prompt_format}', Annotated: {is_annotated}")
        print(f"  Log Directory created: {log_dir}")

        end_info_path = os.path.join(log_dir, "end_info.json")
        initial_data = {
            "metadata": {
                "model": model_name,
                "repeat_actions": repeat_count,
                "levels_folder": levels_folder,
                "timestamp": now_time,
                "prompt_format": prompt_format,
                "annotation_mode": annotation_mode
            }, "results": []
        }
        with open(end_info_path, "w") as f: json.dump(initial_data, f, indent=4)

    qwen_pipe = initialize_qwen_vlm(model_name) if model_name.startswith("Qwen") else None

    run_evaluation_suite(
        levels_folder=levels_folder, log_dir=log_dir, model_name=model_name,
        repeat_count=repeat_count, prompt_format=prompt_format,
        is_annotated=is_annotated, qwen_pipe=qwen_pipe, skip_levels=completed_levels
    )
