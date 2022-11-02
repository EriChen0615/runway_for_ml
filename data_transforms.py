from easydict import EasyDict
from .global_variables import register_to, register_func_to_registry, DataTransform_Registry
from transformers import AutoTokenizer
import copy
import pandas as pd
from torchvision.transforms import ColorJitter, ToTensor
from tqdm import tqdm
from typing import Dict
from collections.abc import Iterable, Mapping
from datasets import Dataset, DatasetDict

def register_transform(fn):
    register_func_to_registry(fn, DataTransform_Registry)
    def _fn_wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return _fn_wrapper

def keep_ds_columns(ds, keep_cols):
    all_colummns = set(ds.features.keys())
    remove_cols = list(all_colummns - set(keep_cols))
    return ds.remove_columns(remove_cols)

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
        register_func_to_registry(self.name, DataTransform_Registry)

    def __apply__(self, data, *args, **kwargs):
        preprocesed_data = self._preprocess(data) # any preprocessing should be handled here
        mapped_data = self._apply_mapping(preprocessed_data, self.input_mapping)
        self._check_input(mapped_data)

        output_data = self._call(**mapped_data) if self.input_mapping else self._call(mapped_data)
        output_mapped_data = self._apply_mapping(output_data, self.output_mapping)
        self._check_output(output_mapped_data)

        return output_mapped_data
        
        # _call will expand keyword arguments from data if mapping [input_col_name : output_col_name] is given
        # otherwise received whole data
    
    def _apply_mapping(self, data, in_out_col_mapping):
        """
        IMPORTANT: when input_mapping is given, data will be transformed into EasyDict
        """
        if in_out_col_mapping is None:
            return data
        assert isinstance(data, Mapping), f"input feature mapping cannot be performed on non-Mapping type objects!"
        mapped_data = {}
        for input_col, output_col in in_out_col_mapping.items():
            mapped_data[output_col] = data[input_col]
        return EasyDict(mapped_data)



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

    def _call(self, *args, **kwargs):
        raise NotImplementedError(f'Must implement {self.name}._call() to be a valid transform')

class RowWiseTransform(BaseTransform):
    """
    Transform each element row-by-row
    """
    def __apply__(self, data, *args, **kwargs):
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
    def _check_input(self, data):
        return isinstance(data, Dataset) or isinstance(data, DatasetDict)
    

    def _apply_mapping(self, data, in_out_col_mapping):
        if in_out_col_mapping is None:
            return data
        if isinstance(data, DatasetDict):
            mapped_data = {out_col_name: data[in_col_name] for in_col_name, out_col_name in in_out_col_mapping.items()}
            return mapped_data
        else: # data is DatasetDict
            data = data.rename_columns(in_out_col_mapping)
            mapped_data = keep_ds_columns(data, list(in_out_col_mapping.values()))
            return mapped_data
    

def multi_feature_row_transform(row_transform_fn):
    """
    NOTE: there is overhead to combine multiple features into a dataframe
    """
    def _transform_wrapper(in_features, *args, **kwargs):
        df = pd.DataFrame(in_features)
        outputs = df.apply(lambda row: row_transform_fn(row, *args, **kwargs))
        output_cols = {feat_name: [] for feat_name in in_features.keys()}
        for output in outputs:
            for col_name in output_cols:
                output_cols[col_name].append(output[col_name])
        res_df = pd.concat([df, pd.DataFrame(output_cols)], ignore_index=True)
        return {col_name: res_df[col_name] for col_name in res_df}
    return _transform_wrapper

def single_feature_row_transform(row_transform_fn):
    """
    Decorator: transforms a single feature row by row
    """
    def _transform_wrapper(in_features, out_features, *args, **kwargs):
        transformed_data = None
        for feat_name, feat_data in in_features.items():
            transformed_data = [row_transform_fn(row, *args, **kwargs) for row in feat_data]
        output = {}
        for col_name in out_features:
            output.update({col_name: [transformed_row[col_name] for transformed_row in transformed_data]})
        return output
        # col_name = out_features[0] if len(out_features) else list(in_features.keys())[0]
        # return {col_name: transformed_data}
    return _transform_wrapper

def ComposeTransforms(transform_fns):
    def _compose_wrapper(inputs, *args, **kwargs):
        res = copy.deepcopy(inputs)
        for t in transform_fns:
            res = t(res, *args, **kwargs)
        return res
    return _compose_wrapper

@register_to(DataTransform_Registry, name="ColorJitterTransform")
@single_feature_row_transform
def ColorJitterTransform(
    image,
    brightness=0.5,
    hue=0.5,
    ):
    func = ColorJitter(brightness=brightness, hue=hue)
    transformed_image = func(image)
    return transformed_image

@register_to(DataTransform_Registry, name="ToTensorTransform")
@single_feature_row_transform
def ToTensorTransform(
    image,
    ):
    func = ToTensor()
    transformed_image = func(image)
    return transformed_image


@register_to(DataTransform_Registry)
def CopyFields( 
    in_features: dict={}, 
    out_features: dict=[]
    ) -> EasyDict:
    '''
    Copy fields directly. If mapping is not specified, the original names will be used
    '''
    output = EasyDict()
    for i, (feat_name, feat_data) in enumerate(in_features.items()):
        out_field = out_features[i]
        output[out_field] = copy.deepcopy(feat_data) # ensure we store data, not reference to data
    return output

@register_to(DataTransform_Registry)
def TokenizeField(
    in_features: dict={}, 
    tokenizer_name: str="null",
    batched=True,
    encode_args: dict={}
    ) -> EasyDict:
    output = EasyDict()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    for feat_name, feat_data in in_features.items():
        encoded_feat = None
        if batched:
            encoded_feat = tokenizer.batch_encode_plus(feat_data, **encode_args)
        else:
            encoded_feat = []
            for row in feat_data:
                encoded_feat.append(tokenizer.encode_plus(row, **encode_args))
        output[feat_name] = encoded_feat
    return output
        
@register_to(DataTransform_Registry)
def MergeFields(
    in_features: Dict={},
    mapping: Dict[str, str]={"in1,in2,in3":"out"}, # input_fields separated by , in the order of merging
    sep_token: str=" ",
    ) -> EasyDict:
    output = EasyDict()
    for in_fields, dest_field in mapping.items():
        in_fields_list = in_fields.split(',')
        feature_length = len(in_features[in_fields_list[0]]) # assuming all feature length are equal
        output_feat = [None] * feature_length
        for i in range(feature_length):
            output_feat[i] = sep_token.join([in_features[in_field] for in_field in in_fields_list])
        output[dest_field] = output_feat
    return output








