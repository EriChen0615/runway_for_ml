from abc import ABC, abstractmethod
from easydict import EasyDict
from collections import defaultdict
import torch

FeatureLoader_Registry = EasyDict() # registry for feature loaders
DataTransform_Registry = EasyDict() # registry for data transforms

def register_to(registry):
    def _register_func(func):
        fn = func.__name__
        registry[fn] = func
        def _func_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return _func_wrapper
    return _register_func

class MapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        use_features,
        ):
        self.data = EasyDict()
        self.use_features = use_features
        self.col_len = 0 
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
        config,
        ):
        self.config = config
        self.in_features = self.config.in_features
        self.transforms = EasyDict(self.config.transforms)
        self.data = defaultdict(EasyDict) # underlying storage class must be enumerable, but otherwise no assumptions
        self.out_data = defaultdict(EasyDict) # container for output data after transformation
        self.dataloader_args = self.config.dataloader_args
        self.result_datasets = EasyDict()
    
    def _assign_to_col(self, split, colname, data):
        self.data[split].colname = data
    
    def _append_to_col(self, split, colname, data):
        for i, row in enumerate(self.data[split].colname):
            row += data[i] # row data structure must support += 
        
    def _select_cols(self, split, colnames):
        out = EasyDict
        for col in colnames:
            out = self.data[split].col
        return out
        
    def load_features(self):
        for in_feature in self.in_features:
            fname = in_feature.feature_name
            fl_name = in_feature.feature_loader.name
            fl_kwargs = in_feature.feature_loader.kwargs
            splits = in_feature.splits
            use_cache = in_feature.use_cache
            for split in splits:
                self.data[split][fname] = FeatureLoader_Registry.fl_name(**fl_kwargs, use_cache=use_cache, split=split)
    
    def apply_transforms(self):
        for split in ['train', 'test', 'valid']:
            for transform in self.transforms.split:
                transform_fn = transform.name
                use_feature_names = transform.use_features
                outputs = DataTransform_Registry.transform_fn(
                    in_features={fname: self.data[split].fname for fname in use_feature_names},
                    **transform.kwargs,
                )
                out_fields = set()
                # note: in-place transformation: self.data is altered. 
                for colname, output_data in outputs.items():
                    if colname[-1] == '+':
                        self._append_to_col(split, colname, output_data)
                        out_fields.add(colname[:-1])
                    else:
                        self._assign_to_col(split, colname, output_data)
                        out_fields.add(colname)
            # only select the columns specified in `out_fields`
            self.out_data[split] = self._select_cols(split, list(out_fields))
            self.result_datasets[split] = MapDataset(self.out_data[split])
    
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
    




            


        
