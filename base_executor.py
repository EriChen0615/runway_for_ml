import pytorch_lightning as pl
from data_modules import DataPipeline
from configuration import (
    TrainingConfig,
    TestingConfig,
    ValidationConfig,
)

class BaseExecutor(pl.LightningModule):
    """
    The class responsible for executing experiments (training, testing, inference, etc.)
    Defines the detail preprocessing/train/test/validation schemes
    """
    def __init__(self,
        input_data_pipeline: DataPipeline, # the input dataset pipeline (without tokenization and model-specific operations)
        model_data_pipeline: DataPipeline, # the data pipeline eventually used by the executor - contains model-specific preprocessing
        model,
        train_config: TrainingConfig,
        test_config: TestingConfig,
        valid_config: ValidationConfig,
        ):
        self.input_data_pipeline = input_data_pipeline
        self.model_data_pipeline = model_data_pipeline
        self.model = model
        self.train_config = train_config
        self.test_config = test_config
        self.valid_config = valid_config
    
    def prepare_data(self):
        """
        tokenization should happen here
        """
        self.input_data_pipeline.run()
        self.model_data_pipeline.run()
    
    def setup(self, stage):
        pass
    
    def configure_optimizers(self):
        """
        Return optimizers and schedulers
        """
        pass

    def train_dataloader(self):
        return self.model_data_pipeline.train_loader()
    
    def val_dataloader(self):
        return self.model_data_pipeline.valid_dataloader()
    
    def test_dataloader(self):
        return self.model_data_pipeline.test_dataloader()
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    