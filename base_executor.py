import pytorch_lightning as pl
from data_modules import DataPipeline
from configuration import (
    DataPipelineConfig,
    ModelConfig,
)

class BaseExecutor(pl.LightningModule):
    """
    The class responsible for executing experiments (training, testing, inference, etc.)
    Defines the detail preprocessing/train/test/validation schemes
    """
    def __init__(self,
        input_data_pipeline_config: DataPipelineConfig,
        model_datapipeline_config: DataPipelineConfig, # the input dataset pipeline (without tokenization and model-specific operations)
        model_config: ModelConfig,
        ):
        self.input_dp_config = input_data_pipeline_config
        self.model_dp_config = model_datapipeline_config
        self.input_data_pipeline = DataPipeline(self.input_dp_config)
        self.model_data_pipeline = DataPipeline(self.model_dp_config)

        self.model_config = model_config
    
    def prepare_data(self):
        """
        tokenization should happen here
        """
        self.model_data_pipeline.run() # self.input_data_pipeline is only called when the transform is required.
    
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


    