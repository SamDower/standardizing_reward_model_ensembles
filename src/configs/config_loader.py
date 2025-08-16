import yaml
import os

def load_config(config_path):
    script_dir = os.path.dirname(__file__)
    abs_config_path = os.path.join(script_dir, f"{config_path}.yaml")

    with open(abs_config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def override_config(base_config, new_config):
    for key, value in new_config.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
            override_config(base_config[key], value)
        else:
            base_config[key] = value

    return base_config
