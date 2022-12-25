import os
import yaml


with open(os.path.join(".", "config.yaml"), "r") as yaml_file:
    yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)


__all__ = ["yaml_data"]
