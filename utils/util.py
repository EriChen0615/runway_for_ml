import transformers

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

