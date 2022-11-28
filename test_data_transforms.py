import os
from .utils.config_system import read_config
from .runway_main import initialize_config, prepare_data
from data_processing import *


def test_data_main(config_file):
    config_dict, meta_config, dp_config = initialize_config(config_file)
    processed_data = prepare_data(dp_config)
    pass

if __name__ == '__main__':
    test_data_main('configs/exp_configs/example_experiment_config.jsonnet')
    

    
