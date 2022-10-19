"""
This file defines the functions that load features. Each function loads several features (columns). 
The return object should `train`, `test`, `valid` keys, corresponding to the three splits 
There are no restrictions on the storage class of feature, as long as the transform understands how to process it.
"""
from runway_for_ml.data_modules import register_to, FeatureLoader_Registry, DataTransform_Registry
from easydict import EasyDict
from collections import defaultdict

@register_to(FeatureLoader_Registry)
def LoadFEATURENAME() -> dict[str, EasyDict]:
    """
    Return a dictionary with three splits (train, test, valid) as keys.
    Each item is a column, It can be any type.
    """
    pass