import logging 
from logging.handlers import RotatingFileHandler
import os
from collections.abc import Iterable, Mapping
import random
random.seed(1018)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DummyBase: pass
class DataPipelineInspector(DummyBase):
    def __init__(self): pass

    def setup_logger(self, log_dir, maxBytes=20000, backupCount=3):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(
            filename=os.path.join(log_dir, f"test-{self.name}.log"),
            maxBytes=maxBytes,
            backupCount=backupCount,
        )
        self.logger.addHandler(handler)
    
    def setup_inspector(self, config_dict):
        self.do_inspect = True
        self.setup_logger(config_dict['log_dir'])
        

    def inspect_transform_before(self, transformation_name, transform, outputs):
        self.logger.info(f"{transformation_name}")
        transform_fn = transform.name
        in_col_mapping = transform.in_col_mapping
        out_col_mapping = transform.out_col_mapping
        self.logger.info(f"Transform name: {transform_fn}\nin_col_mapping: {in_col_mapping}\nout_col_mapping: {out_col_mapping}")
        pass

    def inspect_transform_after(self, transformation_name, transform, outputs):
        pass

    def inspect_loaded_features(self, data):
        self.logger.info(f"Loaded data: {data}")
        if isinstance(data, Iterable):
            indices = random.randint(0, len(data))
            for i in indices:
                self.logger.info(f"Index={i}: {data[i]}")
        pass
    

class TestClass(DummyBase):
    def __init__(self, name):
        self.name = name
    
    def run(self):
        if hasattr(self, 'do_inspect'):
            self.log_info('hello! This is '+self.name)




if __name__ == '__main__':
    t1 = TestClass('Eric')   
    extend_instance(t1, DataPipelineInspector)
    t1.setup_logger()
    t1.run()

    
    
