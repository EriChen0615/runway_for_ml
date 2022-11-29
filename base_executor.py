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
    
    # def _init_tokenizer(self, tokenizer_config):
    #     tokenizer_class_name = tokenizer_config.class_name
    #     tokenizer_version_name = tokenizer_config.version_name
    #     TokenizerClass = getattr(transformers, tokenizer_class_name)
    #     self.tokenizer = TokenizerClass.from_pretrained(tokenizer_version_name)


    def prepare_data(self):
        """
        tokenization should happen here
        """
        self.model_data_pipeline.run() # self.input_data_pipeline is only called when the transform is required.
    
    def setup(self, stage):
        self.pipeline_output_data = self.model_data_pipeline.output_data
    
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
        return self.model_data_pipeline.train_dataloader()
    
    def val_dataloader(self):
        return self.model_data_pipeline.valid_dataloader()
    
    def test_dataloader(self):
        return self.model_data_pipeline.test_dataloader()
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    