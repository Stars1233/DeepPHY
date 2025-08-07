# deepphy/planner/prompt_strategy_03_kinetix.py

def generate_prompts_and_images(num_active_joints, num_active_thrusters,
                                previous_action_list, distance_list, image_list,
                                current_step, current_image_paths, last_action,
                                prompt_format, is_annotated):
    """
    Generates system prompt, user prompt, and image descriptions for the LLM.
    Supports VLA/WM formats and annotated/unannotated images. All text is in English.

    Args:
        ...
        current_image_paths (dict): A dict with paths, e.g., {'unannotated': 'path', 'annotated': 'path'}.
        prompt_format (str): 'VLA' or 'WM'.
        is_annotated (bool): True if using annotated images.
    """
    total_actions = num_active_joints + num_active_thrusters
    motor_labels = ", ".join([f"M{i+1}" for i in range(num_active_joints)])
    thruster_labels = ", ".join([f"T{i+1}" for i in range(num_active_thrusters)])

    # --- System Prompt Generation ---
    if is_annotated:
        system_prompt_text = (
            f"You are an AI agent controlling entities in a 2D physics environment.\n"
            f"Your task is to generate an integer vector of length {total_actions} to control the labeled entities.\n"
            f"You will be given two images of the current scene: one clean image and one with labels (M=Motor, T=Thruster).\n\n"
            f"Action Vector Structure:\n"
            f"- The first {num_active_joints} elements correspond to motors, labeled {motor_labels}.\n"
            f"  - `0`: No action, `1`: Positive rotation, `2`: Negative rotation\n"
            f"- The next {num_active_thrusters} elements correspond to thrusters, labeled {thruster_labels}.\n"
            f"  - `0`: No action, `1`: Positive thrust\n\n"
        )
    else: # Unannotated
        system_prompt_text = (
            f"You are an AI agent controlling entities in a 2D physics environment.\n"
            f"Your primary task is to generate an integer vector of total length {total_actions}.\n"
            f"This vector is structured as follows:\n"
            f"1.  The first {num_active_joints} elements represent joint actions.\n"
            f"    *   Possible values for joint actions: `0` (No action), `1` (Positive rotation), `2` (Negative rotation)\n"
            f"2.  The subsequent {num_active_thrusters} elements represent thruster actions.\n"
            f"    *   Possible values for thruster actions: `0` (No action), `1` (Positive thrust)\n\n"
        )

    if prompt_format == 'VLA':
        system_prompt_text += (
            f"Your output MUST consist ONLY of this vector. Do NOT include any explanations.\n"
            f"Objective: Make green objects touch blue objects, AND green objects MUST NOT touch red objects."
        )
    elif prompt_format == 'WM':
        system_prompt_text += (
            f"Your output MUST be in the following two-part format:\n"
            f"Prediction: Your prediction of the environmental changes after executing the action.\n"
            f"Action: Your action vector as a formatted list of integers, e.g. `[action_M1, ..., action_T1, ...]`\n"
            f"Objective: Make green objects touch blue objects, AND green objects MUST NOT touch red objects."
        )

    # --- User Prompt and Image Descriptions ---
    image_descriptions = []
    processed_previous_actions = []
    for item in previous_action_list:
        item_list = item.tolist()
        processed_action = item_list[0:num_active_joints] + item_list[4:4+num_active_thrusters]
        processed_previous_actions.append(processed_action)

    num_history_steps = len(image_list)
    if num_history_steps == 0:
        history_text = f"This is your first action (Step {current_step}). There is no prior history."
    else:
        history_text = (
            f"This is your action for Step {current_step}. "
            f"Below is the history of the previous {num_history_steps} step(s): "
            f"the visual scenes, their corresponding actions {processed_previous_actions}, "
            f"and the resulting object distances {distance_list}."
        )
        image_descriptions.append({'path': image_list[0], 'label': history_text})
        for i, img_path in enumerate(image_list[1:], start=2):
            label = "History annotated scene {i}." if is_annotated else "History scene {i}."
            image_descriptions.append({'path': img_path, 'label': label})

    processed_last_action_list = last_action.tolist()
    processed_last_action = processed_last_action_list[0:num_active_joints] + processed_last_action_list[4:4+num_active_thrusters]

    # --- Current Scene Text and Images ---
    unannotated_path = current_image_paths['unannotated']
    annotated_path = current_image_paths.get('annotated') # Will be None if not annotated

    if current_step == 1:
        if is_annotated:
            current_scene_text = "Below is the initial visual scene, followed by the same scene with controllable entities labeled. Decide your first action."
        else:
            current_scene_text = "Below is the initial visual scene. Decide your first action."

        # Combine history intro text with the first image label
        first_label = f"{history_text} {current_scene_text}" if num_history_steps == 0 else current_scene_text
        image_descriptions.append({'path': unannotated_path, 'label': first_label})

    else: # Subsequent steps
        if is_annotated:
            current_scene_text = f"The last action taken was {processed_last_action}. Below is the current visual scene, followed by the labeled version."
        else:
            current_scene_text = f"The last action taken was {processed_last_action}. Below is the current visual scene."
        image_descriptions.append({'path': unannotated_path, 'label': current_scene_text})

    if is_annotated and annotated_path:
        image_descriptions.append({'path': annotated_path, 'label': "This is the annotated version. Use the labels (M, T) to map your actions."})

    # --- User Prompt Text ---
    user_prompt_text = (
        "Goal: Make green objects touch blue objects; Green objects must NOT touch red objects. "
        f"Based on the provided information, output your response for Step {current_step} NOW. "
    )
    if prompt_format == 'VLA':
        user_prompt_text += f"Your entire response must be ONLY the {total_actions}-dimensional action vector."
    elif prompt_format == 'WM':
        user_prompt_text += "Your response MUST strictly follow the 'Prediction: ... Action: ...' format."

    return system_prompt_text, user_prompt_text, image_descriptions