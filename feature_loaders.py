from data_modules import register_to, FeatureLoader_Registry, DataTransform_Registry
from datasets import load_dataset
from easydict import EasyDict
from collections import defaultdict

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
