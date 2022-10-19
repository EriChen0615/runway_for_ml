from data_modules import register_to, FeatureLoader_Registry, DataTransform_Registry
from datasets import load_dataset
from easydict import EasyDict
from collections import defaultdict
from datasets import load_dataset, Image

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
def LoadBeansDataset():
    return LoadHFDataset("beans", ['image', 'labels'])
    # beans_ds = load_dataset("beans")
    # ds = defaultdict(EasyDict)
    # for split in ['train', 'test', 'validation']:
    #     ds[split] = beans_ds[split]
    # ds['valid'] = ds['validation']
    # return ds

def LoadHFDataset(dataset_name, fields=[]):
    hf_ds = load_dataset(dataset_name)
    ds = defaultdict(EasyDict)
    for split in ['train', 'test', 'validation']:
        if len(fields) == 0:
            ds[split] = hf_ds[split]
        else:
            all_columns = set(hf_ds[split].features.keys())
            keep_fields = set(fields)
            remove_fields = all_columns - keep_fields
            ds[split] = hf_ds[split].remove_columns(list(remove_fields))
    ds['valid'] = ds.pop('validation')
    return ds
