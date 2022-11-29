import pytorch_lightning as pl
from .data_pipeline import DataPipeline
from .configuration import (
    DataPipelineConfig,
    ModelConfig,
)
import transformers
from transformers import AdamW, Adafactor, get_scheduler

class BaseExecutor(pl.LightningModule):
    """
    The class responsible for executing experiments (training, testing, inference, etc.)
    Defines the detail preprocessing/train/test/validation schemes
    """
    def __init__(self,
        data_pipeline_config: DataPipelineConfig,
        model_config: ModelConfig,
        inference_config,
        mode, # train/infer/eval
        ):
        self.dp_config = data_pipeline_config
        self.dp = DataPipeline(self.input_dp_config)

        self.model_config = model_config
        self.optimizer_config = model_config.optimizer_config
        self.training_config = model_config.train 
        self.additional_kwargs = model_config.additional_kwargs
        
        self.mode = mode

        self._init_model(self.model_config)
        self.save_hyperparameters()

    
    def _init_model(self, model_config: ModelConfig):
        ModelClass = getattr(globals()[model_config.ModelLib], model_config.ModelClass)
        if model_config.ModelLib == 'transformers':
            if model_config.checkpoint_path:
                self.model = ModelClass.from_pretrained(
                    model_config.checkpoint_path, 
                    **model_config.load_checkpoint_kwargs)
            else:
                self.model = ModelClass.from_pretrained(
                    model_config.model_version,
                    **model_config.load_checkpoint_kwargs)
        else:
            raise NotImplementedError('The _init_model() method is not defined for library ' + model_config.ModelLib)
    
    def prepare_data(self):
        self.dp.run()
    
    def setup(self, stage):
        """
        Set up self.train_dataset, self.test_dataset and self.val_dataset etc.
        """
        pass
    
    def configure_optimizers(self):
        """
        Return optimizers and schedulers
        """
        optimizer_name = self.optimizer_config['optimizer_name']
        if optimizer_name == 'AdamW':
            optimizer = AdamW(self.parameters(), **self.optimizer_params)
        elif optimizer_name == 'Adafactor':
            optimizer = Adafactor(self.parameters(), **self.optimizer_params)
        else:
            raise ValueError(f"Invaild optimizer name: {self.optimizer}")
        return optimizer

    def train_dataloader(self):
        self.train_dataset.set_format('torch')
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.training_config['batch_size'],
            num_workers=self.training_config.get('dataloader_workers', 8)
        )

    def val_dataloader(self):
        self.val_dataset.set_format('torch')
        return DataLoader(
            self.val_dataset,
            shuffle=True,
            batch_size=self.training_config['batch_size'],
            num_workers=self.training_config.get('dataloader_workers', 8)
        )
    
    def test_dataloader(self):
        self.test_dataset.set_format('torch')
        return DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=self.inference_config['batch_size'],
            num_workers=self.inference_config.get('dataloader_workers', 8)
        )
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    