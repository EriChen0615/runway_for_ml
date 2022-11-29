from abc import ABC, abstractmethod
from easydict import EasyDict
from collections import defaultdict
from .configuration import DataPipelineConfig
import torch
from typing import Union, List, Dict, Optional
from .utils.cache_system import cache_data_to_disk, load_data_from_disk, cache_file_exists, make_cache_file_name
import os 
from tqdm import tqdm
from .global_variables import register_to, DataTransform_Registry
from .data_transforms import *

class DummyBase(object): pass
class DataPipeline(DummyBase):
    def __init__(
        self, 
        config: DataPipelineConfig,
        ):
        self.config = config

        self.name = self.config.name
        self.cache_dir = self.config.cache_dir

        self.transforms = EasyDict(self.config.transforms)

        self.output_data = defaultdict(EasyDict) # container for output data after transformation
        self.output_cache = {}

        # Datapipeline also registered as a feature loader to compose more complex pipelines

        self.logger = None # placeholder for inspector
    
    def _read_from_cache(self, trans_id):
        data = load_data_from_disk(trans_id, self.cache_dir)
        return data 

    def _save_to_cache(self, trans_id, data):
        cache_data_to_disk(data, trans_id, self.cache_dir)
    
    def _check_cache_exist(self, trans_id):
        cache_file_name = make_cache_file_name(trans_id, self.cache_dir)
        return cache_file_exists(cache_file_name)


    def _exec_transform(self, trans_id):
        # parse transform info
        trans_type, trans_name = trans_id.split(':')
        trans_info = self.transforms[trans_id]

        # Read from cache or disk when available
        if trans_id in self.output_cache:
            print(f"Load {trans_id} from program cache")
            return self.output_cache[trans_id]
        # Read from disk when instructed and available
        elif not trans_info.get('regenerate', True) and self._check_cache_exist(trans_id):
            print(f"Load {trans_id} from disk cache")
            outputs = self._read_from_cache(trans_id)
            self.output_cache[trans_id] = outputs
            return outputs

        print("Execute Transform")
        # Initialize functor
        func = DataTransform_Registry[trans_info.transform_name]()
        func.setup(**trans_info.setup_kwargs)


        # Get input_data
        input_data = None
        if trans_type != "input" and trans_info['input_node']:
            input_trans_id = trans_info['input_node']
            input_data = self._exec_transform(input_trans_id)

        if hasattr(self, 'inspect_transform_before') and self.transforms[trans_id].get('inspect', True): # inspector function
            self.inspect_transform_before(trans_id, self.transforms[trans_id], input_data)

        output = func(input_data)
    
        if hasattr(self, 'inspect_transform_before') and self.transforms[trans_id].get('inspect', True): # inspector function
            self.inspect_transform_after(trans_id, self.transforms[trans_id], output)

        # Cache data if appropriate
        self.output_cache[trans_id] = output
        if trans_info.get('cache', False):
            self._save_to_cache(trans_id, output)
        return output

    def apply_transforms(self):
        for trans_id in self.transforms:
            trans_type, trans_name  = trans_id.split(':')
            if trans_type == 'output':
                self.output_data[trans_name] = self._exec_transform(trans_id) 
        return self.output_data
    
    def get_data(self, out_transforms, explode=False):
        if explode:
            assert len(out_transforms)==1, "To explode data, only one field can be selected"
            return self._exec_transform(out_transforms[0])
        return EasyDict({
            out_trans: self._exec_transform(out_trans)
                for out_trans in out_transforms
        })
        

            