import pytorch_lightning as pl
from data_modules import DataPipeline

class Executor(pl.LightningModule):
    def __init__(self,
        data_pipeline: DataPipeline,
        model,
        ):
        self.data_pipeline = data_pipeline
        self.model = model
    
    def prepare_data(self):
        """
        tokenization should happen here
        """
        pass
    
    def setup(self, stage):
        pass
    
    def configure_optimizers(self):
        """
        Return optimizers and schedulers
        """


    def train_dataloader(self):
        return self.data_pipeline.train_loader()
    
    def val_dataloader(self):
        return self.data_pipeline.valid_dataloader()
    
    def test_dataloader(self):
        return self.data_pipeline.test_dataloader()
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    