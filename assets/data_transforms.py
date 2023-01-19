"""
This file defines the data transforms that will be applied to the data. 
Each transform takes in an EasyDict object of in_features (key: feature_name, value: feature data)
It should output an EasyDict object of out_features (key: feature_name, value: feature_data)
Each transform defined here can be used as an independent unit to form a data pipeline
Some common transforms are provided by runway
"""
from runway_for_ml.data_modules import register_to, FeatureLoader_Registry, DataTransform_Registry
from runway_for_ml.data_ops.data_transforms import *
from easydict import EasyDict

@register_to(DataTransform_Registry)
def YourTransform(
    in_features: EasyDict=None,
    ) -> EasyDict:
    """
    Process the in_features 
    """
    pass