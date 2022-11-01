from easydict import EasyDict
from .global_variables import register_to, register_func_to_registry, DataTransform_Registry
from transformers import AutoTokenizer
import copy
import pandas as pd
from torchvision.transforms import ColorJitter, ToTensor
from tqdm import tqdm
from typing import Dict
from collections.abc import Iterable
from datasets import Dataset, DatasetDict

def register_transform(fn):
    register_func_to_registry(fn, DataTransform_Registry)
    def _fn_wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return _fn_wrapper

class BaseTransform():
    """
    Most general functor definition
    """
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        register_func_to_registry(self.name, DataTransform_Registry)

    def __apply__(self, data, *args, **kwargs):
        preprocesed_data = self._preprocess(data)
        self._check_input(preprocesed_data)
        return self._call(preprocesed_data)

    def _check_input(self, data):
        """
        Check if the transformed can be applied on data. Override in subclasses
        No constraints by default
        """
        return True
    
    def _preprocess(self, data):
        """
        Preprocess data for transform.
        """
        return data

    def _call(self, data, *args, **kwargs):
        raise NotImplementedError(f'Must implement {self.name}._call(data) to be a valid transform')

class RowWiseTransform(BaseTransform):
    """
    Transform each element row-by-row
    """
    def __init__(self, name=None):
        super().__init__(name=name)
    
    def __apply__(self, data, *args, **kwargs):
        for row_n, row_data in enumerate(data):
            self._call(row_data, row_n)
    
    def _call(self, row_data, row_n):
        raise NotImplementedError(f'Must implement {self.name}._call(row_data, row_n) to be a valid transform')

    def _check_input(self, data):
        return isinstance(data, Iterable)

class HFDatasetTransform(BaseTransform):
    """
    Transform using HuggingFace Dataset utility
    """
    def __init__(self, name=None):
        super().__init__(name=name)
    
    def _check_input(self, data):
        return isinstance(data, Dataset) or isinstance(data, DatasetDict)
        


    

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








