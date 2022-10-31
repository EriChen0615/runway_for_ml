from abc import ABC, abstractmethod
from easydict import EasyDict
from collections import defaultdict
from .configuration import DataPipelineConfig
import torch
from typing import Union, List, Dict, Optional
from .utils.cache_system import cache_data_to_disk, load_data_from_disk, cache_file_exists, make_cache_file_name
import os 
from tqdm import tqdm
from .global_variables import register_to, DataTransform_Registry, FeatureLoader_Registry
from .data_transforms import *
from .feature_loaders import *

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
        in_data: Dict[str, any],
        use_features: List[str] = [], # use all by default
        ):
        self.data = EasyDict()
        self.use_features = use_features if len(use_features) else list(in_data.keys()) # by default, use all features
        self.col_len = 0 
        for col_name in self.use_features:
            self.data[col_name] = in_data[col_name]
            if self.col_len == 0:
                self.col_len = len(self.data[col_name])
            else:
                assert self.col_len == len(self.data[col_name]), "all features (columns) must be of the same length"

    def __len__(self):
        return self.col_len

    def __getitem__(self, idx):
        return {feature: self.data[feature][idx] for feature in self.use_features}


def _select_cols(in_data, colnames):
    out = EasyDict()
    for col in colnames:
        out[col] = in_data[col]
    return out

def _prepare_to_tensor(arr):
    #TODO: Make configurable!
    if isinstance(arr, torch.Tensor):
        pass # already a tensor
    if isinstance(arr, list) and not isinstance(arr[0], torch.Tensor):
        return torch.Tensor(arr)
    elif isinstance(arr, list) and isinstance(arr[0], torch.Tensor):
        return torch.stack(arr)
    else:
        raise NotImplementedError("Fail to prepare to tensor to collate samples")

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
        self.dataloaders_use_features = self.config.dataloaders_use_features

        self.data = defaultdict(EasyDict) # underlying storage class must be enumerable, but otherwise no assumptions
        self.output_ok_flag = False
        self.output_data = defaultdict(EasyDict) # container for output data after transformation
        self.result_datasets = None

        # Datapipeline also registered as a feature loader to compose more complex pipelines
        self.this_loader_name = f"Load{self.name}" 
        assert self.this_loader_name not in FeatureLoader_Registry, f"Identical Feature Loaders names detected, '{self.this_loader_name}' is already used!"
        FeatureLoader_Registry[self.this_loader_name] = self._LoadThisDataPipeline # register corresponding FeatureLoader

    def _LoadThisDataPipeline(self, split='train'):
            output_data = self.run()
            return output_data[split]

    def _assign_to_col(self, split, colname, in_data, out_data):
        # self.data[split].colname = data
        out_data[split][colname] = in_data

    
    def _append_to_col(self, split, colname, in_data, out_data):
        for i, row in enumerate(out_data[split].colname):
            row += in_data[i] # row data structure must support += 
        
    # def _select_cols(self, split, colnames):
    #     out = EasyDict()
    #     for col in colnames:
    #         out = self.data[split][col]
    #     return out

    def load_features(self):
        for in_feature in self.in_features:
            feature_names = in_feature.feature_names
            fl_name = in_feature.feature_loader.name
            fl_kwargs = in_feature.feature_loader.kwargs
            splits = in_feature.splits
            use_cache = in_feature.use_cache
            for split in splits:
                loaded_data = FeatureLoader_Registry[fl_name](**fl_kwargs, split=split)
                for feat_name in feature_names:
                    self.data[split][feat_name] = loaded_data[feat_name]
    
    def apply_transforms(self):
        for split in ['train', 'test', 'valid']:
            pbar = tqdm(self.transforms[split], unit='op')
            outputs = self.data[split]
            for transform in pbar:
                pbar.set_description(f"{self.name}-{split}-{transform.name}")
                transform_fn = transform.name
                use_feature_names = transform.use_features
                out_feature_names = transform.out_features
                outputs.update(
                    DataTransform_Registry[transform_fn](
                    in_features={fname: outputs[fname] for fname in use_feature_names},
                    out_features=out_feature_names,
                    **transform.kwargs)
                )
                pass
                #NOTE: in-place transformation: self.data is altered. 
                out_fields = set(outputs.keys())
                # for colname, output_data in outputs.items():
                #     if colname[-1] == '+':
                #         # self._append_to_col(split, colname, output_data)
                #         out_fields.add(colname[:-1])
                #     else:
                #         # self._assign_to_col(split, colname, output_data)
                #         out_fields.add(colname)
            # only select the columns specified in `out_fields`
            self.output_data[split] = _select_cols(outputs, list(out_fields))
        self.output_ok_flag = True
            # self.output_data[split] = outputs
        # self._convert_out_data_to_datasets(self.dataloaders_use_features)
        
    def _convert_out_data_to_datasets(self, dataloaders_use_features): # make MapDataset with split_name as key
        self.result_datasets = EasyDict()
        for split in ['train', 'test', 'valid']:
            self.result_datasets[split] = MapDataset(self.output_data[split], use_features=dataloaders_use_features[split]) # all
    
    def run(self):
        """
        Run the data pipeline. Load cache data or apply transform. 
        If in_data is not None, use it as input data for transforms
        Return self.out_data & self.result_datasets
        """
        if self.output_ok_flag:
            return self.output_data
        if self.regenerate or not cache_file_exists(self.cache_file_name):
            self.load_features()
            self.apply_transforms()
            if self.cache_data:
                cache_data_to_disk(self.output_data, self.name, self.cache_dir)
        else:
            self.output_data = load_data_from_disk(self.name, self.cache_dir)
            self.output_ok_flag = True
        return self.output_data
    
    def make_collate_fn(self, split):
        batched_data = {feat_name: [] for feat_name in self.dataloaders_use_features[split]}
        def _collate_fn(examples): # Let user do it.
            for example in examples:
                for feat_name, feat_data in example.items():
                    batched_data[feat_name].append(feat_data)
            for feat_name in batched_data:
                batched_data[feat_name] = _prepare_to_tensor(batched_data[feat_name]) #TODO: leave this to user
            return batched_data
        return _collate_fn
        
    def train_dataloader(self):
        if self.result_datasets is None:
            self._convert_out_data_to_datasets(self.dataloaders_use_features)
        collate_fn = self.make_collate_fn('train')
        return torch.utils.data.DataLoader(
            self.result_datasets['train'],
            **self.config.dataloader_args['train'],
            collate_fn=collate_fn,
        )


    def test_dataloader(self):
        if self.result_datasets is None:
            self._convert_out_data_to_datasets(self.dataloaders_use_features)
        collate_fn = self.make_collate_fn('test')
        return torch.utils.data.DataLoader(
            self.result_datasets['test'],
            **self.config.dataloader_args['test'],
            collate_fn=collate_fn,
        )

    def valid_dataloader(self):
        if self.result_datasets is None:
            self._convert_out_data_to_datasets(self.dataloaders_use_features)
        collate_fn = self.make_collate_fn('valid')
        return torch.utils.data.DataLoader(
            self.result_datasets['valid'],
            **self.config.dataloader_args['valid'],
            collate_fn=collate_fn
        )
    




            


        
