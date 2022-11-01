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
    
    def __init__(self, *args, **kwargs):
        self.from_config(*args, **kwargs)

    def __post_init__(self):
    # Loop through the fields
        for field in fields(self):
            # If there is a default and the value of the field is none we can assign a value
            if not isinstance(field.default, dataclasses._MISSING_TYPE) and getattr(self, field.name) is None:
                setattr(self, field.name, None)

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
    dataloaders_use_features: Dict[str, List[str]] = None

    def from_config(self, config: Dict[str, any], meta_config: Dict[str, any], key_name: str = 'data_pipeline'):
        config_dict = config[key_name]
        self.name = config_dict['name'] if 'name' in config_dict else "DefaultDataPipeline"
        self.in_features = config_dict['in_features']
        self.transforms = EasyDict(config_dict['transforms'])
        self.dataloader_args = EasyDict(config_dict['dataloader_args'])
        self.cache_dir = config_dict['cache_dir'] if 'cache_dir' in config_dict else meta_config.default_cache_dir
        self.regenerate = config_dict.get('regenerate', True)
        self.cache_data = config_dict.get('cache_data', True)
        self.dataloaders_use_features = config_dict.get('dataloaders_use_features')
        self.do_inspect = config_dict.get('do_inspect', False)
        self.inspector_config = config_dict.get('inspector_config', None)



@dataclass
class ModelConfig(ConfigClass):
    model_name: str = None
    model_lib: str = None # [HF | torch ...]
    ModelClass: str = None
    ModelConfigClass: str = None
    model_config_args: Dict[str, any] = None
    tokenize: bool = False
    TokenizerClass: str = None
    tokenizer_args: Dict[str, any] = None
    # TokenizerModelVersion: str = None
    special_tokens: List[Dict[str, any]] = None
    # input_transforms: List[any] = None
    # decode_input_transforms: List[any] = None
    # output_transforms: List[any] = None

    def from_config(self, config: Dict[str, any]):
        config_dict = config.model
        self.model_name = config_dict['model_name'] if 'model_name' in config_dict else "DefaultModel"
        self.model_lib = config_dict['model_lib'] if 'model_lib' in config_dict else "HF"

        self.ModelClass = config_dict['ModelClass']
        self.ModelConfigClass = config_dict['ModelConfigClass']
        self.model_config_args = config_dict.get('model_config_args', None)

        # for NLP models, tokenizers are required
        self.needs_tokenizer = config_dict.get('needs_tokenizer', False)
        self.TokenizerClass = config_dict.get('TokenizerClass', None)
        self.tokenizer_args = config_dict.get('tokenizer_args', None)
        self.special_tokens = config_dict.get('special_tokens', [])

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





    