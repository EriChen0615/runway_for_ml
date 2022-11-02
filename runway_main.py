from .data_modules import DataPipeline
from .configuration import (
    MetaConfig, 
    DataPipelineConfig, 
    # ModelConfig, 
    # LoggingConfig, 
    # TrainingConfig, 
    # ValidationConfig, 
    # TestingConfig
)
import os
from .utils.config_system import read_config
from .utils.mixin_utils import extend_instance
from .inspectors import DataPipelineInspector
from easydict import EasyDict


# meta_config = MetaConfig()
# dp_config = DataPipelineConfig() # data pipeline config
# next_dp_config = DataPipelineConfig() # next data pipeline config

# CONFIG_FILE = os.path.join('configs', 'exp_configs', 'example_experiment_config.jsonnet')

def initialize_config(config_file):
    config_dict = read_config(config_file)
    meta_config = MetaConfig.from_config(config_dict)  
    dp_config = DataPipelineConfig.from_config(config_dict, meta_config)
    # next_dp_config.from_config(config, meta_config, key_name="next_data_pipeline")
    return (
        config_dict,
        meta_config,
        dp_config,
    )

def prepare_data(dp_config: DataPipelineConfig):
    data_pipeline = DataPipeline(dp_config)
    if dp_config.do_inspect:
        extend_instance(data_pipeline, DataPipelineInspector)
        data_pipeline.setup_inspector(dp_config.inspector_config)
    processed_data = data_pipeline.run()
    return processed_data


if __name__ == '__main__':
    # config_dict = initialize_config(CONFIG_FILE)
    # processed_data = prepare_data(config_dict)
    # train_dataloader = processed_data.train_dataloader()
    # print(next(iter(train_dataloader)))
    pass