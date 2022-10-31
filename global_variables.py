from easydict import EasyDict
import functools

FeatureLoader_Registry = EasyDict() # registry for feature loaders
DataTransform_Registry = EasyDict() # registry for feature loaders
Model_Registry = EasyDict()

def register_to(registry, name=None):
    def _register_func(func):
        fn = name or func.__name__
        registry[fn] = func
        @functools.wraps(func)
        def _func_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return _func_wrapper
    return _register_func