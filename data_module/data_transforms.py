from easydict import EasyDict
from ..utils.global_variables import register_to, register_func_to_registry, DataTransform_Registry
from transformers import AutoTokenizer
import transformers
import copy
import pandas as pd
from torchvision.transforms import ColorJitter, ToTensor
from tqdm import tqdm
from typing import Dict, List
from collections.abc import Iterable, Mapping
from datasets import Dataset, DatasetDict, load_dataset
import functools


def register_transform(fn):
    register_func_to_registry(fn, DataTransform_Registry)
    def _fn_wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return _fn_wrapper

def keep_ds_columns(ds, keep_cols):
    all_colummns = set(ds.features.keys())
    remove_cols = list(all_colummns - set(keep_cols))
    return ds.remove_columns(remove_cols)

def register_transform_functor(cls):
    register_func_to_registry(cls, DataTransform_Registry)
    return cls

class BaseTransform():
    """
    Most general functor definition
    """
    def __init__(
        self,
        *args,
        name=None,
        input_mapping: Dict=None,
        output_mapping: Dict=None,
        **kwargs
        ):
        self.name = name or self.__class__.__name__
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping

    # @classmethod 
    # def __init__subclass__(cls, **kwargs):
    #     super().__init_subclass__(*args, **kwargs)
    #     register_func_to_registry(cls.__name__, DataTransform_Registry)

    def __call__(self, data, *args, **kwargs):
        preprocessed_data = self._preprocess(data) # any preprocessing should be handled here
        # mapped_data = self._apply_mapping(preprocessed_data, self.input_mapping)
        self._check_input(preprocessed_data)

        # output_data = self._call(**mapped_data) if self.input_mapping else self._call(mapped_data)
        output_data = self._call(preprocessed_data)
        # output_mapped_data = self._apply_mapping(output_data, self.output_mapping)
        self._check_output(output_data)

        return output_data
        
        # _call will expand keyword arguments from data if mapping [input_col_name : output_col_name] is given
        # otherwise received whole data
    
    # def _apply_mapping(self, data, in_out_col_mapping):
    #     """
    #     IMPORTANT: when input_mapping is given, data will be transformed into EasyDict
    #     """
    #     if in_out_col_mapping is None:
    #         return data
    #     assert isinstance(data, Mapping), f"input feature mapping cannot be performed on non-Mapping type objects!"
    #     mapped_data = {}
    #     for input_col, output_col in in_out_col_mapping.items():
    #         mapped_data[output_col] = data[input_col]
    #     return EasyDict(mapped_data)



    def _check_input(self, data):
        """
        Check if the transformed can be applied on data. Override in subclasses
        No constraints by default
        """
        return True
    
    def _check_output(self, data):
        """
        Check if the transformed data fulfills certain conditions. Override in subclasses
        No constraints by default
        """
        return True
        
    
    def _preprocess(self, data):
        """
        Preprocess data for transform.
        """
        return data

    def setup(self, *args, **kwargs):
        """
        setup any reusable resources for the transformed. Will be called before __apply__()
        """
        raise NotImplementedError(f"Must implement {self.name}.setup() to be a valid transform")

    def _call(self, data, *args, **kwargs):
        raise NotImplementedError(f'Must implement {self.name}._call() to be a valid transform')

class RowWiseTransform(BaseTransform):
    """
    Transform each element row-by-row
    """
    # @classmethod 
    # def __init__subclass__(cls, **kwargs):
    #     super().__init_subclass__(*args, **kwargs)
    #     register_func_to_registry(cls.__name__, DataTransform_Registry)

    def __call__(self, data, *args, **kwargs):
        preprocesed_data = self._preprocess(data) # any preprocessing should be handled here
        self._check_input(preprocesed_data)
        for row_n, row_data in enumerate(preprocesed_data):
            mapped_data = self._apply_mapping(row_data, self.input_mapping)
            output_data = self._call(row_n, **mapped_data) if self.input_mapping else self._call(row_n, mapped_data)
            output_mapped_data = self._apply_mapping(output_data, self.output_mapping)
        self._check_output(output_mapped_data)
        return output_mapped_data

    def _call(self, row_n, row_data):
        raise NotImplementedError(f'Must implement {self.name}._call() to be a valid transform')

    def _check_input(self, data):
        return isinstance(data, Iterable)

class HFDatasetTransform(BaseTransform):
    """
    Transform using HuggingFace Dataset utility
    """
    # @classmethod 
    # def __init__subclass__(cls, **kwargs):
    #     super().__init_subclass__(*args, **kwargs)
    #     register_func_to_registry(cls.__name__, DataTransform_Registry)
    def setup(self, rename_col_dict, *args, **kwargs):
        """
        setup any reusable resources for the transformed. Will be called before __call__()
        For HFDataset, add rename_col_dict for renaming columns conveniently
        """
        self.rename_col_dict = rename_col_dict

    def _check_input(self, data):
        return isinstance(data, Dataset) or isinstance(data, DatasetDict)
    
    # def _apply_mapping(self, data, in_out_col_mapping):
    #     if not in_out_col_mapping:
    #         return data
    #     if isinstance(data, DatasetDict):
    #         mapped_data = {out_col_name: data[in_col_name] for in_col_name, out_col_name in in_out_col_mapping.items()}
    #         return mapped_data
    #     else: # data is DatasetDict
    #         data = data.rename_columns(in_out_col_mapping)
    #         mapped_data = keep_ds_columns(data, list(in_out_col_mapping.values()))
    #         return mapped_data
    
def tokenize_function(tokenizer, field, **kwargs):
    def tokenize_function_wrapped(example):
        return tokenizer.batch_encode_plus(example[field], **kwargs)
    return tokenize_function_wrapped

@register_transform_functor
class HFDatasetTokenizeTransform(HFDatasetTransform):
    def setup(self, rename_col_dict, tokenizer_config: EasyDict, tokenize_fields_list: List):
        super().setup(rename_col_dict)
        self.tokenize_fields_list = tokenize_fields_list
        self.version_name = tokenizer_config.version_name
        self.class_name = tokenizer_config.class_name
        self.special_tokens = tokenizer_config.get('special_tokens', {})
        self.tokenize_kwargs = tokenizer_config.get(
            'tokenize_kwargs', 
            {
             'batched': True,
             'load_from_cache_file': False,
             'padding': 'max_length',
             'truncation': True
             }
        )

        TokenizerClass = getattr(transformers, self.class_name)
        self.tokenizer = TokenizerClass.from_pretrained(self.version_name)

        if self.class_name[:4] == 'GPT2':
            self.tokenizer.pad_token = '[PAD]'
        self.tokenizer.add_special_tokens(self.special_tokens)

    def _call(self, dataset):
        results = {}
        for split in ['train', 'test', 'validation']:
            # ds = dataset[split].select((i for i in range(100)))
            ds = dataset[split]
            for field_name in self.tokenize_fields_list:
                ds = ds\
                .map(tokenize_function(self.tokenizer, field_name, **self.tokenize_kwargs), batched=True, load_from_cache_file=False) \
                .rename_columns({
                    'input_ids': field_name+'_input_ids',
                    'attention_mask': field_name+'_attention_mask',
                })
            ds = ds.rename_columns(self.rename_col_dict)
            results[split] = ds
        return results

@register_transform_functor
class LoadHFDataset(BaseTransform):
    def setup(self, dataset_path, dataset_name, fields=[]):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.fields = fields
    
    def _call(self, data):
        hf_ds = load_dataset(self.dataset_path, self.dataset_name, cache_dir='./cache/')
        return hf_ds

