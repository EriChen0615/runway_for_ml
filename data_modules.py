from abc import ABC, abstractmethod
from easydict import EasyDict
from collections import defaultdict

FeatureLoader_Registry = EasyDict() # registry for feature loaders
DataTransform_Registry = EasyDict() # registry for data transforms

def register_to(registry):
    def _register_func(func):
        fn = func.__name__
        registry[fn] = func
        def func_wrapper(*args, **kwargs):
            return func_wrapper(*args, **kwargs)
        return func_wrapper
    return _register_func

class DataPipeline(ABC):
    def __init__(
        self, 
        config,
        ):
        self.config = config
        self.in_features = self.config.in_features
        self.transforms = EasyDict(self.config.transforms)
        self.data = defaultdict(EasyDict) # underlying storage class must be enumerable, but otherwise no assumptions
    
    def _new_col(self, split, colname, data):
        self.data[split].colname = data
    
    def _append_to_col(self, split, colname, data):
        for i, row in enumerate(data[split].colname):
            row += data[i] # row data structure must support += 

    
    def load_features(self):
        for in_feature in self.in_features:
            fname = in_feature.feature_name
            fl_name = in_feature.feature_loader.name
            fl_args = in_feature.feature_loader.args
            splits = in_feature.splits
            use_cache = in_feature.use_cache
            for split in splits:
                self.data[split][fname] = FeatureLoader_Registry.fl_name(**fl_args, use_cache=use_cache, split=split)
    
    def apply_transforms(self):
        for split in ['train', 'test', 'valid']:
            for transform in self.transforms.split:
                transform_fn = transform.name
                use_feature_names = transform.use_features
                outputs = DataTransform_Registry.transform_fn(
                    **transform.args
                    **{fname: self.data[split].fname for fname in use_feature_names},
                )
                for colname, output_data in outputs.items():
                    if colname[-1] == '+':
                        self._append_to_col(split, colname, output_data)
                    else:
                        self._new_col(split, colname, output_data)
    
    # Collate fn? Check weizhe's code
    @abstractmethod
    def train_dataloader(self):
        pass

    @abstractmethod
    def test_dataloader(self):
        pass

    @abstractmethod
    def valid_dataloader(self):
        pass
    




            


        
