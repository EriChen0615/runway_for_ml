from data_modules import DataPipeline
from configuration import (
    MetaConfig, 
    DataPipelineConfig, 
    ModelConfig, 
    LoggingConfig, 
    TrainingConfig, 
    ValidationConfig, 
    TestingConfig
)
import os
from utils.config_system import read_config


meta_config = MetaConfig()
dp_config = DataPipelineConfig() # data pipeline config

CONFIG_FILE = os.path.join('configs', 'exp_configs', 'example_experiment_config.jsonnet')
if __name__ == '__main__':
    config = read_config(CONFIG_FILE)
    meta_config.from_config(config)  
    dp_config.from_config(config, meta_config)
    print(dp_config)
    pass