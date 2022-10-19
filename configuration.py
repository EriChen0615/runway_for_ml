"""
This file defines the config classes. The configuration file (jsonnet) will be 
converted into these config classes before they are used by the code.

Each class should implement the from_config() method, which initialize the class
members from the loaded config dict.
"""

from dataclasses import dataclass, fields
import dataclasses
from easydict import EasyDict
from abc import ABC, abstractmethod
from typing import Dict, List
from pprint import pprint

@dataclass
class ConfigClass:
    """
    Base class for all config classes. Have additional field for flexibility
    """
    additional: Dict[str, any] = None
    
    def __post_init__(self):
    # Loop through the fields
        for field in fields(self):
            # If there is a default and the value of the field is none we can assign a value
            if not isinstance(field.default, dataclasses._MISSING_TYPE) and getattr(self, field.name) is None:
                setattr(self, field.name, field.default)

    @abstractmethod
    def from_config(self, config: Dict[str, any], meta_config: Dict[str, any]=None):
        raise NotImplementedError


@dataclass
class MetaConfig(ConfigClass):
    wandb_cache_dir: str = ""
    wandb_user_name: str = ""
    wandb_project_name: str = ""
    default_cache_dir: str = ""
    data_folder: str = ""
    seed: int = 2022
    platform_type: str = "pytorch"
    cuda: int = 0
    gpu_device: int = 0

    def from_config(self, config: Dict[str, any], meta_config: Dict[str, any]=None):
        config_dict = config['meta_config']
        pprint(config_dict)
        self.default_cache_dir = config_dict['default_cache_dir']


@dataclass
class DataPipelineConfig(ConfigClass):
    name: str = ""
    in_features: List[Dict[str, any]] = None 
    transforms: Dict[str, Dict[str, any]] = None # [split - [key - value]]
    dataloader_args: Dict[str, Dict[str, any]] = None # [split - [arg_name - arg_value]]
    cache_dir: str = ""
    regenerate: bool = True

    def from_config(self, config: Dict[str, any], meta_config: Dict[str, any]):
        config_dict = config.data_pipeline
        self.name = config['name'] if 'name' in config else "DefaultDataPipeline"
        self.in_features = config_dict['in_features']
        self.transforms = EasyDict(config_dict['transforms'])
        self.dataloader_args = EasyDict(config_dict['dataloader_args'])
        self.cache_dir = config_dict['cache_dir'] if 'cache_dir' in config_dict else meta_config.default_cache_dir
        self.regenerate = config_dict.get('regenerate') or True
        self.cache_data = config_dict.get('cache_data') or True


# @dataclass
# class ModelConfig(ConfigClass):
#     base_model: str
#     ModelClass: str
#     TokenizerClass: str
#     TokenizerModelVersion: str
#     ConfigClass: str
#     special_tokens: List[Dict[str, any]]
#     input_transforms: List[any]
#     decode_input_transforms: List[any]
#     output_transforms: List[any]


# @dataclass
# class LoggingConfig(ConfigClass):
#     name: str

# @dataclass
# class TrainingConfig(ConfigClass):
#     executor_class: str
#     epochs: int
#     batch_size: int
#     lr: float
#     scheduler: str
#     additional: Dict[str, any]

# @dataclass
# class ValidationConfig(ConfigClass):
#     batch_size: int
#     step_size: int
#     break_interval: int
#     additional: Dict[str, any]

# @dataclass
# class TestingConfig(ConfigClass):
#     evaluation_name: str
#     load_epoch: int
#     load_model_path: str
#     load_best_model: bool
#     batch_size: int
#     num_evaluation: int
#     additional: Dict[str, any]
#     metrics: List[Dict[str, str]]





    