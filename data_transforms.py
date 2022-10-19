from easydict import EasyDict
from data_modules import register_to, FeatureLoader_Registry, DataTransform_Registry
from transformers import AutoTokenizer
import copy

@register_to(DataTransform_Registry)
def CopyFields( 
    in_features=None, 
    mapping=None
    ) -> EasyDict:
    '''
    Copy fields directly. If mapping is not specified, the original names will be used
    '''
    output = EasyDict()
    for feat_name, feat_data in in_features.items():
        out_field = feat_name
        if feat_name in mapping:
            out_field = mapping[feat_name]
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
    in_features: dict={},
    mapping: dict[str, str]={"in1,in2,in3":"out"}, # input_fields separated by , in the order of merging
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








