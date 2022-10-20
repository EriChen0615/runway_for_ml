"""
This is the file for feature loaders. The feature loader must accept `use_cache` and `split` arguments
"""
from data_modules import register_to, FeatureLoader_Registry, DataTransform_Registry
from datasets import load_dataset
from easydict import EasyDict
from collections import defaultdict
from datasets import load_dataset, Image

def LoadHFDataset(dataset_name, split='train', fields=[]):
    if split == 'valid':
        split = 'validation'
    hf_ds = load_dataset(dataset_name, split=split)
    ds = defaultdict(EasyDict)
    if len(fields) == 0:
        ds = hf_ds
    else:
        all_columns = set(hf_ds.features.keys())
        keep_fields = set(fields)
        remove_fields = all_columns - keep_fields
        ds = hf_ds.remove_columns(list(remove_fields))
    return ds

@register_to(FeatureLoader_Registry)
def LoadSGDWithSchema():
    dataset = defaultdict(EasyDict)
    for split in ['train', 'test', 'validation']:
        sgd_split = load_dataset('gem', 'schema_guided_dialog', split=split)
        for field in ['dialog_acts', 'service', 'prompt', 'target', 'references']:
            dataset[split][field] = sgd_split[field] 
        dataset[split]["general_prompt"] = f"This is a template for {split}"
    dataset['valid'] = dataset['validation']
    return dataset

@register_to(FeatureLoader_Registry)
def LoadBeansDataset(use_cache=True, split='train'):
    return LoadHFDataset("beans", fields=['image', 'labels'], split=split)
    # beans_ds = load_dataset("beans")
    # ds = defaultdict(EasyDict)
    # for split in ['train', 'test', 'validation']:
    #     ds[split] = beans_ds[split]
    # ds['valid'] = ds['validation']
    # return ds


