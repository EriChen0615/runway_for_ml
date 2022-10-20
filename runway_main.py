from data_modules import DataPipeline
from configuration import (
    MetaConfig, 
    DataPipelineConfig, 
    # ModelConfig, 
    # LoggingConfig, 
    # TrainingConfig, 
    # ValidationConfig, 
    # TestingConfig
)
import os
from utils.config_system import read_config


meta_config = MetaConfig()
dp_config = DataPipelineConfig() # data pipeline config

CONFIG_FILE = os.path.join('configs', 'exp_configs', 'example_experiment_config.jsonnet')

def initialize_config(config_file):
    config = read_config(config_file)
    meta_config.from_config(config)  
    dp_config.from_config(config, meta_config)

def prepare_data():
    data_pipeline = DataPipeline(dp_config)
    processed_data = data_pipeline.run()
    return data_pipeline


if __name__ == '__main__':
    initialize_config(CONFIG_FILE)
    processed_data = prepare_data()
    pass