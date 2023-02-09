import transformers
import torch

def get_tokenizer(tokenizer_config):
    tokenizer_dict = tokenizer_config
    tokenizer_name = tokenizer_dict['version_name']
    tokenizer_class = tokenizer_dict['class_name']
    tokenizer_class_obj = getattr(transformers, tokenizer_class)
    tokenizer = tokenizer_class_obj.from_pretrained(tokenizer_name)
    special_tokens = {}
    if tokenizer_class[:4] == 'GPT2':
        # special_tokens.update({'pad_token': '[PAD]'})
        tokenizer.pad_token = '[PAD]'
        # tokenizer.pad_token_id = tokenizer.eos_token_id
    special_tokens.update(tokenizer_dict.get('special_tokens', {}))
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def batch_depad(x, attension_mask=None, pad_len=0):
    max_in_length = torch.max((~(x==0)).sum(dim=-1))+pad_len
    attension_mask = attension_mask[:, :max_in_length]
    return x[:, :max_in_length], attension_mask

