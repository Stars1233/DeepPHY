import os
import json
from dotenv import load_dotenv

from deepphy.utils import Singleton

load_dotenv(verbose=True)

class Config(metaclass=Singleton):

    def __init__(self):
        pass

    def load_env_config(self, config_json_path):

        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        full_config_json_path = os.path.join(self.project_root, config_json_path)

        try:
            with open(full_config_json_path, 'r') as f:
                json_data = json.load(f)

            for key, value in json_data.items():
                setattr(self, key, value)

        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {full_config_json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Config file format error: {full_config_json_path}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the config file: {e}")
