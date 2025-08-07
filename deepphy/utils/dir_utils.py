import os
import datetime

def create_main_results_dir(log_dir):
    """
    A helper function to create a timestamped results directory.
    This function is not core to the tree display logic but is kept as it was in your code.
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    main_dir = os.path.join(log_dir, timestamp)
    os.makedirs(main_dir, exist_ok=True)
    print(f"Main results directory created: {main_dir}")
    return main_dir
