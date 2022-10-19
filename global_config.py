import json
from utils.config_system import process_config
from dataclasses import dataclass


def initialize_global_config(config_file):
    config = process_config(config_file)