import pytorch_lightning as pl
from ..data_module.data_pipeline import DataPipeline
from ..configs.configuration import (
    DataPipelineConfig,
    ModelConfig,
)
import transformers
from transformers import AdamW, Adafactor, get_scheduler
from torch.utils.data import DataLoader

class BaseExecutor(pl.LightningModule):
    """
    The class responsible for executing experiments (training, testing, inference, etc.)
    Defines the detail preprocessing/train/test/validation schemes
    """
    def __init__(self,
        data_pipeline_config: DataPipelineConfig,
        model_config: ModelConfig,
        mode, # train/infer/eval
        train_config={},
        infer_config={},
        *args, **kwargs
        ):
        super().__init__()
        self.dp_config = data_pipeline_config
        self.dp = DataPipeline(self.dp_config)

        self.model_config = model_config
        self.optimizer_config = train_config.optimizer_config
        self.training_config = train_config
        self.additional_kwargs = model_config.additional_kwargs
        
        self.mode = mode

        self._init_model(self.model_config)
        self.save_hyperparameters()

    
    def _init_model(self, model_config: ModelConfig):
        ModelClass = getattr(globals()[model_config.ModelLib], model_config.ModelClass)
        if model_config.ModelLib == 'transformers': # transformer models
            if model_config.get('checkpoint_path', None):
                self.model = ModelClass.from_pretrained(
                    model_config.checkpoint_path, 
                    **model_config.load_checkpoint_kwargs)
            else:
                self.model = ModelClass.from_pretrained(
                    model_config.model_version,
                    **model_config.load_checkpoint_kwargs)
        elif model_config.ModelLib == 'modeling': # custom models
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
        self.dp.apply_transforms()
    
    def setup(self, stage):
        """
        Set up self.train_dataset, self.test_dataset and self.val_dataset etc.
        """
        raise NotImplementedError('Need to implement setup()') 
    
    def configure_optimizers(self):
        """
        Return optimizers and schedulers
        """
        optimizer_name = self.optimizer_config['optimizer_name']
        optimizer_params = self.optimizer_config.get('optimizer_params', {})
        if optimizer_name == 'AdamW':
            optimizer = AdamW(self.parameters(), **optimizer_params)
        elif optimizer_name == 'Adafactor':
            optimizer = Adafactor(self.parameters(), **optimizer_params)
        else:
            raise ValueError(f"Invaild optimizer name: {optimizer_name}")
        return optimizer

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.training_config['batch_size'],
            num_workers=self.training_config.get('dataloader_workers', 8)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.training_config['batch_size'],
            num_workers=self.training_config.get('dataloader_workers', 8)
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=self.inference_config['batch_size'],
            num_workers=self.inference_config.get('dataloader_workers', 8)
        )
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
