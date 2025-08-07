import sys
import os
import time
import math
import pyautogui
import subprocess

def get_window_geometry_macos(app_name):
    try:
        script = f'''
            tell application "System Events"
                tell process "{app_name}"
                    if not (exists front window) then return ""
                    set frontWindow to front window
                    set windowPosition to position of frontWindow
                    set windowSize to size of frontWindow
                    return {{item 1 of windowPosition, item 2 of windowPosition, item 1 of windowSize, item 2 of windowSize}}
                end tell
            end tell
        '''
        result = subprocess.run(
            ['osascript', '-e', script],
            check=True, capture_output=True, text=True, timeout=1 # Add timeout to prevent hangs
        )
        output = result.stdout.strip()
        if not output: # AppleScript returns an empty string if the app window doesn't exist
            return None

        coords = [int(n) for n in output.split(', ')]

        if len(coords) == 4:
            return tuple(coords)
        else:
            print(f"Error: AppleScript returned an unexpected format: {output}")
            return None
    except subprocess.CalledProcessError as e:
        # This usually happens if the app is not open or the name is wrong;
        # checked at startup, so we can handle it silently during runtime.
        return None
    except Exception as e:
        print(f"An unknown error occurred while parsing window information: {e}")
        return None

def restart_mac_app(app_name: str):
    """
    Completely quits and reopens a specified application on macOS.

    :param app_name: The exact name of the application (e.g., "Safari", "Calculator", "WeChat").
    """
    print(f"--- Starting to restart application: {app_name} ---")

    # 1. Quit the application (using AppleScript)
    print(f"Step 1: Attempting to quit '{app_name}'...")
    try:
        # Build the AppleScript command
        # 'tell application "AppName" to quit' is a standard AppleScript statement
        script = f'tell application "{app_name}" to quit'

        # Execute the osascript command using subprocess.run
        # check=True would throw an exception if the command fails
        # capture_output=True and text=True capture output for debugging
        result = subprocess.run(
            ['osascript', '-e', script],
            check=False,  # Set to False, so the script doesn't stop if the app isn't running
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"'{app_name}' has been sent the quit request successfully.")
        else:
            # If the return code is not 0, the app might not have been open
            if "is not running" in result.stderr:
                print(f"Info: '{app_name}' was not running initially.")
            else:
                print(f"Encountered an error while quitting '{app_name}': {result.stderr.strip()}", file=sys.stderr)

    except Exception as e:
        print(f"An unknown error occurred while executing the quit command: {e}", file=sys.stderr)
        return # If quitting fails, do not proceed

    # 2. Reopen the application (using the 'open' command)
    print(f"\nStep 2: Attempting to reopen '{app_name}'...")
    try:
        # 'open -a AppName' is the standard command to open an app by name on macOS
        subprocess.run(['open', '-a', app_name], check=True)
        print(f"'{app_name}' has been launched successfully.")
    except FileNotFoundError:
        print(f"Error: The 'open' command was not found. Please ensure you are running this on macOS.", file=sys.stderr)
    except subprocess.CalledProcessError:
        print(f"Error: Could not find an application named '{app_name}'. Please check the name.", file=sys.stderr)
    except Exception as e:
        print(f"An unknown error occurred while reopening the app: {e}", file=sys.stderr)

    print(f"\n--- Application '{app_name}' restart complete ---")

def perform_action_drag(app_name, start_coords, end_coords):
    """Performs a drag action from a relative start point to a relative end point."""
    if not activate_app(app_name): return False
    geometry = get_window_geometry_macos(app_name)
    if not geometry: return False

    win_left, win_top, win_width, win_height = geometry

    # Convert relative start and end points to absolute screen coordinates
    start_x_abs = win_left + win_width * start_coords['x']
    start_y_abs = win_top + win_height * start_coords['y']
    end_x_abs = win_left + win_width * end_coords['x']
    end_y_abs = win_top + win_height * end_coords['y']

    print(f"    - Performing Action: 'DRAG CUT' from ({start_coords['x']:.3f}, {start_coords['y']:.3f}) to ({end_coords['x']:.3f}, {end_coords['y']:.3f})")

    # Execute the mouse actions
    pyautogui.moveTo(start_x_abs, start_y_abs, duration=0.1)
    pyautogui.mouseDown(button='left')
    time.sleep(0.05)
    pyautogui.dragTo(end_x_abs, end_y_abs, duration=0.2, button='left', _pause=False)
    pyautogui.mouseUp(button='left')

    print(f"    - [+] Drag cut completed.")
    return True

def perform_cut_circle(app_name, circle_info, radius, circles=1):
    """Performs a circular cut action."""
    if not activate_app(app_name): return False
    geometry = get_window_geometry_macos(app_name)
    if not geometry: return False

    win_left, win_top, win_width, win_height = geometry
    center_x_abs = win_left + win_width * circle_info['x']
    center_y_abs = win_top + win_height * circle_info['y']
    radius_abs = win_width * radius

    print(f"    - Performing Action: 'MAX SPEED CIRCLE CUT' at ({circle_info['x']:.3f}, {circle_info['y']:.3f})")
    num_steps, total_angle = int(25 * circles), 360 * circles
    path = []
    for i in range(num_steps + 1):
        angle = math.radians((i / num_steps) * total_angle)
        path.append((center_x_abs + radius_abs * math.cos(angle), center_y_abs + radius_abs * math.sin(angle)))

    pyautogui.moveTo(path[0][0], path[0][1], duration=0.05)
    pyautogui.mouseDown(button='left')
    time.sleep(0.05)
    for x, y in path[1:]:
        pyautogui.dragTo(x, y, duration=0, button='left', _pause=False)
    pyautogui.mouseUp(button='left')
    print(f"    - [+] Circle cut completed.")
    return True

def activate_app(app_name):
    """Activates the specified application window."""
    try:
        script = f'tell application "{app_name}" to activate'
        subprocess.run(['osascript', '-e', script], check=True, capture_output=True, text=True, timeout=5)
        time.sleep(0.5)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print(f"[!] Error: Could not activate '{app_name}'.")
        return False

def take_window_screenshot(app_name, screenshot_dir, filename):
    """Takes a screenshot of the application window."""
    geometry = get_window_geometry_macos(app_name)
    if not geometry:
        print(f"    - [!] Warning: Could not get window geometry for screenshot.")
        return
    left, top, width, height = geometry
    try:
        os.makedirs(screenshot_dir, exist_ok=True)
        filepath = os.path.join(screenshot_dir, filename)
        pyautogui.screenshot(region=(left, top, width, height)).save(filepath)
        print(f"    - [+] Screenshot saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"    - [!] Error: Failed to save screenshot. Reason: {e}")
        return None

def perform_action_at_relative_coords(app_name, action_info):
    """Performs a UI action at relative coordinates."""
    if not activate_app(app_name): return False
    geometry = get_window_geometry_macos(app_name)
    if not geometry: return False

    win_left, win_top, win_width, win_height = geometry
    abs_x = win_left + win_width * action_info['x']
    abs_y = win_top + win_height * action_info['y']
    action_type = action_info.get('action', 'click')

    print(f"    - Performing Action: '{action_type.upper()}' at relative ({action_info['x']:.3f}, {action_info['y']:.3f})")
    pyautogui.moveTo(abs_x, abs_y, duration=0.1)
    if action_type == "click":
        pyautogui.click()
    elif action_type == "double_click":
        pyautogui.doubleClick()
    elif action_type == "long_press":
        pyautogui.mouseDown()
        time.sleep(action_info.get('duration', 0.2))
        pyautogui.mouseUp()
    return True

def perform_action_swipe(app_name, direction='left'):
    """Performs a swipe left action."""

    pyautogui.mouseUp()

    action_info = {"x": 0.5, "y": 0.3}
    width = 0.025
    if not activate_app(app_name): return False
    geometry = get_window_geometry_macos(app_name)
    if not geometry: return False

    win_left, win_top, win_width, win_height = geometry
    abs_x = win_left + win_width * action_info['x']
    abs_y = win_top + win_height * action_info['y']
    swipe_distance = win_width * width

    print(f"    - Performing Action: 'SWIPE' '{direction.upper()}'")
    pyautogui.moveTo(abs_x, abs_y, duration=0.1)
    pyautogui.mouseDown()
    time.sleep(0.1)
    if direction == 'left':
        pyautogui.moveRel(-swipe_distance, 0, duration=0.2)
    elif direction == 'right':
        pyautogui.moveRel(swipe_distance, 0, duration=0.2)
    pyautogui.mouseUp()
    return True

def perform_action_shoot(app_name, start_point, angle_degrees, power_ratio):
    """
    Performs a slingshot action, like in Angry Birds.

    It simulates dragging the slingshot from a starting point, pulling it back
    based on an angle and power, and then releasing it. The pull-back coordinates
    are clamped to stay within the application window boundaries.

    :param app_name: Name of the application window.
    :param start_point: A dictionary with relative coordinates of the slingshot,
                        e.g., {'x': 0.25, 'y': 0.65}.
    :param angle_degrees: The desired launch angle in degrees. 0 is horizontal to the right,
                          positive angles are upwards, negative are downwards.
    :param power_ratio: The power of the shot, from 0.0 (no pull) to 1.0 (max pull).
    """
    if not activate_app(app_name): return False
    geometry = get_window_geometry_macos(app_name)
    if not geometry: return False

    win_left, win_top, win_width, win_height = geometry

    # Ensure power is within the valid range [0, 1]
    power_ratio = max(0.0, min(1.0, power_ratio))

    # Convert relative start point to absolute screen coordinates
    start_x_abs = win_left + win_width * start_point['x']
    start_y_abs = win_top + win_height * start_point['y']

    # Define a maximum pull distance (e.g., 20% of the window width). This is a key parameter to tune.
    max_pull_distance_abs = win_width * 0.2
    pull_distance_abs = max_pull_distance_abs * power_ratio

    # Calculate the pull-back vector based on standard mathematical angles
    # (0 degrees right, 90 degrees up). We add 180 degrees (pi radians) to get the pull-back direction.
    pull_angle_rad = math.radians(angle_degrees)
    delta_x = pull_distance_abs * math.cos(pull_angle_rad + math.pi)
    delta_y = pull_distance_abs * math.sin(pull_angle_rad + math.pi)

    # Calculate the initial absolute end point for the mouse drag
    initial_end_x_abs = start_x_abs + delta_x
    # The y-axis needs to be inverted for screen coordinates (where y increases downwards)
    initial_end_y_abs = start_y_abs - delta_y

    # --- MODIFICATION START: Clamp coordinates to window boundaries ---
    win_x_min, win_x_max = win_left, win_left + win_width
    win_y_min, win_y_max = win_top, win_top + win_height

    end_x_abs = max(win_x_min, min(initial_end_x_abs, win_x_max-25))
    end_y_abs = max(win_y_min, min(initial_end_y_abs, win_y_max-25))
    # --- MODIFICATION END ---

    print(f"    - Performing Action: 'SHOOT' from ({start_point['x']:.2f}, {start_point['y']:.2f}) with angle={angle_degrees}, power={power_ratio:.2f}")
    print(f"      - Start (abs): ({start_x_abs:.0f}, {start_y_abs:.0f})")

    # If clamping occurred, show a more detailed log
    if int(end_x_abs) != int(initial_end_x_abs) or int(end_y_abs) != int(initial_end_y_abs):
        print(f"      - Pull (raw):  ({initial_end_x_abs:.0f}, {initial_end_y_abs:.0f}) -> CLAMPED")
        print(f"      - Pull (final):  ({end_x_abs:.0f}, {end_y_abs:.0f})")
    else:
        print(f"      - Pull (abs):    ({end_x_abs:.0f}, {end_y_abs:.0f})")


    # Execute the mouse actions to simulate the shot
    pyautogui.moveTo(start_x_abs, start_y_abs, duration=0.2)
    pyautogui.mouseDown(button='left')
    time.sleep(0.1)
    pyautogui.dragTo(end_x_abs, end_y_abs, duration=1, button='left')
    pyautogui.mouseUp(button='left')

    print(f"    - [+] Shoot action completed.")
    return True
