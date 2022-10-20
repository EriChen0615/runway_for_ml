from abc import ABC, abstractmethod
from easydict import EasyDict
from collections import defaultdict
from configuration import DataPipelineConfig
import torch
from typing import Union, List, Dict, Optional
from utils.cache_system import cache_data_to_disk, load_data_from_disk, cache_file_exists, make_cache_file_name
import os 
from tqdm import tqdm

FeatureLoader_Registry = EasyDict() # registry for feature loaders
DataTransform_Registry = EasyDict() # registry for feature loaders

def register_to(registry, name=None):
    def _register_func(func):
        fn = name or func.__name__
        registry[fn] = func
        def _func_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return _func_wrapper
    return _register_func

from data_transforms import *
from feature_loaders import *

# def cache_feature_loader():
#     def _loader_func(load_func):
#         fname = load_func.__name__
#         def _func_wrapper(*args, **kwargs):
#             if kwargs.get('use_cache') and cache_file_exists():
#                 pass # use cache
#             else:
#                 return load_func(*args, **kwargs)
#         return _func_wrapper
#         if kwargs.get('use')
#     return _loader_func

class MapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: Dict[str, any],
        use_features: List[str] = [],
        ):
        self.data = EasyDict()
        self.use_features = use_features
        self.col_len = 0 
        if use_features == []:
            for feature, col in data.items():
                self.data[feature] = col
                if self.col_len == 0:
                    self.col_len = len(col)
                else:
                    assert self.col_len == len(col), "all features (columns) must be of the same length"
        else: # use_features is a list
            for feature in self.use_features:
                self.data[feature] = data.feature
                if self.col_len == 0:
                    self.col_len = len(self.data[feature])
                else:
                    assert self.col_len == len(self.data[feature]), "all features (columns) must be of the same length"
    
    def __len__(self):
        return self.col_len

    def __getitem__(self, idx):
        return {feature: self.data.feature for feature in self.use_features}


class DataPipeline:
    def __init__(
        self, 
        config: DataPipelineConfig,
        ):
        self.config = config

        self.name = self.config.name
        self.cache_dir = self.config.cache_dir
        self.cache_file_name = make_cache_file_name(self.name, self.cache_dir)
        self.regenerate = self.config.regenerate
        self.cache_data = self.config.cache_data

        self.dataloader_args = self.config.dataloader_args
        self.in_features = self.config.in_features
        self.transforms = EasyDict(self.config.transforms)

        self.data = defaultdict(EasyDict) # underlying storage class must be enumerable, but otherwise no assumptions
        self.output_data = defaultdict(EasyDict) # container for output data after transformation
        self.result_datasets = EasyDict()
    
    def _assign_to_col(self, split, colname, data):
        self.data[split].colname = data
    
    def _append_to_col(self, split, colname, data):
        for i, row in enumerate(self.data[split].colname):
            row += data[i] # row data structure must support += 
        
    def _select_cols(self, split, colnames):
        out = EasyDict()
        for col in colnames:
            out = self.data[split][col]
        return out
        
    def load_features(self):
        for in_feature in self.in_features:
            feature_names = in_feature.feature_names
            fl_name = in_feature.feature_loader.name
            fl_kwargs = in_feature.feature_loader.kwargs
            splits = in_feature.splits
            use_cache = in_feature.use_cache
            for split in splits:
                loaded_data = FeatureLoader_Registry[fl_name](**fl_kwargs, use_cache=use_cache, split=split)
                for feat_name in feature_names:
                    self.data[split][feat_name] = loaded_data[feat_name]
    
    def apply_transforms(self):
        for split in ['train', 'test', 'valid']:
            outputs = None
            pbar = tqdm(self.transforms[split])
            for transform in pbar:
                pbar.set_description(f"{split}-{transform.name}")
                transform_fn = transform.name
                use_feature_names = transform.use_features
                outputs = DataTransform_Registry[transform_fn](
                    in_features={fname: self.data[split][fname] for fname in use_feature_names},
                    **transform.kwargs,
                )
                #NOTE: in-place transformation: self.data is altered. 
                # out_fields = set()
                #TODO: implement '+' operation
                # for colname, output_data in outputs.items():
                #     if colname[-1] == '+':
                #         self._append_to_col(split, colname, output_data)
                #         out_fields.add(colname[:-1])
                #     else:
                #         self._assign_to_col(split, colname, output_data)
                #         out_fields.add(colname)
            # only select the columns specified in `out_fields`
            # self.output_data[split] = self._select_cols(split, list(out_fields))
            self.output_data[split] = outputs
        self._convert_out_data_to_datasets()
        
    def _convert_out_data_to_datasets(self): # make MapDataset with split_name as key
        for split in ['train', 'test', 'valid']:
            self.result_datasets[split] = MapDataset(self.output_data[split], use_features=[]) # all
    
    def run(self):
        """
        Run the data pipeline. Load cache data or apply transform. 
        Return self.out_data
        """
        if self.regenerate or not cache_file_exists(self.cache_file_name):
            self.load_features()
            self.apply_transforms()
            if self.cache_data:
                cache_data_to_disk(self.output_data, self.name, self.cache_dir)
        else:
            self.output_data = load_data_from_disk(self.name, self.cache_dir)
        self._convert_out_data_to_datasets()
        return self.output_data
            
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.result_datasets['train'],
            **self.config.dataloader_args['train'],
        )


    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.result_datasets['test'],
            **self.config.dataloader_args['test'],
        )

    def valid_dataloader(self):
        return torch.utils.data.DataLoader(
            self.result_datasets['valid'],
            **self.config.dataloader_args['valid'],
        )
    




            


        
