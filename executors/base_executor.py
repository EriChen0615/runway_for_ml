import pytorch_lightning as pl
from ..data_module.data_pipeline import DataPipeline
from ..configs.configuration import (
    DataPipelineConfig,
    ModelConfig,
)
import transformers
from transformers import AdamW, Adafactor, get_scheduler
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from ..utils.metrics_log_callback import MetricsHistoryLogger
import logging
from ..utils.eval_recorder import EvalRecorder
import os
import copy
logger = logging.getLogger(__name__)

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
        test_config={},
        use_data_node=None,
        log_file_path=None,
        eval_pipeline_config: DataPipelineConfig=None,
        global_config=None,
        eval_recorder_config={},
        *args, **kwargs
        ):
        super().__init__()
        self.dp_config = data_pipeline_config
        self.dp = DataPipeline(self.dp_config, global_config=global_config)
        self.eval_dp_config = eval_pipeline_config
        if self.eval_dp_config is not None:
            self.eval_pipeline = DataPipeline(self.eval_dp_config)
        else:
            self.eval_pipeline = None

        self.model_config = model_config
        self.optimizer_config = train_config.get('optimizer_config', None)
        self.training_config = train_config
        self.test_config = test_config
        self.additional_kwargs = model_config.get("additional_kwargs", {})
        
        self.mode = mode
        self.log_file_path = log_file_path
        self.log_list  = []
        self.test_cnt = 0
        self.valid_cnt = 0

        self.global_config = global_config
        self.config = global_config # Easier to use
        self.use_data_node = use_data_node
        
        self._init_model(self.model_config)

        self.use_wandb = False
        for trainer_logger in kwargs.get('logger', []):
            if type(trainer_logger) == TensorBoardLogger:
                self.tb_logger = trainer_logger
            elif type(trainer_logger) == WandbLogger:
                self.use_wandb = True
                self.wandb_logger = trainer_logger
                self.wandb_logger.watch(self.model, log_freq=500, log_graph=False)
            else:
                logger.warning(f'Unsupported logger type: {type(trainer_logger)}')
        
        
        self.save_hyperparameters()

    
    def _init_model(self, model_config: ModelConfig):
        """Initialize self.model with model_config. 
        Initialization procedure depends on which `ModelLib` is used.
        Implementation available for
        - transformers (using `from_pretrained`)

        Args:
            model_config (ModelConfig): model configurations

        Raises:
            NotImplementedError: Raise error when the `ModelLib` is not supported.
        """
        ModelClass = getattr(globals()[model_config.ModelLib], model_config.ModelClass)
        if model_config.ModelLib == 'transformers': # transformer models
            if model_config.get('train_from_scratch', None):
                ConfigClass = getattr(globals()[model_config.ModelLib], model_config.ConfigClass)
                config_obj = ConfigClass.from_pretrained(model_config.model_version)
                self.model = ModelClass(config_obj) # init from config
            if model_config.get('checkpoint_path', None):
                self.model = ModelClass.from_pretrained(
                    model_config.checkpoint_path, 
                    **model_config.load_checkpoint_kwargs)
            else:
                self.model = ModelClass.from_pretrained(
                    model_config.model_version,
                    **model_config.load_checkpoint_kwargs)
        elif model_config.ModelLib == 'models': # custom models
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
        if self.use_data_node:
            self.prepared_data = self.dp.get_data([self.use_data_node], explode=True)

    
    def setup(self, stage):
        """
        Set up self.train_dataset, self.test_dataset and self.val_dataset etc.
        """
        for trainer_logger in self.trainer.loggers:
            if type(trainer_logger) == TensorBoardLogger:
                self.tb_logger = trainer_logger
            elif type(trainer_logger) == WandbLogger:
                self.wandb_logger = trainer_logger
                self.wandb_logger.watch(self.model, log_freq=500, log_graph=False)
            elif type(trainer_logger) == MetricsHistoryLogger:
                self.metrics_history_logger = trainer_logger
            else:
                logger.warning(f'Unsupported logger type: {type(trainer_logger)}')
        

    def configure_optimizers(self):
        """
        Return optimizers and schedulers
        """
        optimizer_name = self.optimizer_config['optimizer_name']
        optimizer_params = self.optimizer_config.get('optimizer_params', {})

        optimization_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()],
                'lr': optimizer_params.lr,
                'initial_lr': optimizer_params.lr,
            },
        ]
        
        for group in optimization_parameters:
            logger.info('#params: {}   lr: {}'.format(len(group['params']), group['lr']))
        
        """define optimizer"""
        
        if optimizer_name == 'AdamW':
            self.optimizer = AdamW(optimization_parameters, **optimizer_params)
        elif optimizer_name == 'Adafactor':
            self.optimizer = Adafactor(optimization_parameters, **optimizer_params)
        elif optimizer_name == 'Adam':
            self.optimizer = Adam(optimization_parameters, **optimizer_params)
        else:
            raise ValueError(f"Invaild optimizer name: {optimizer_name}")
        
        num_warmup_steps = self.optimizer_config.get('scheduler_params', {}).get('num_warmup_steps', 0)
        if self.optimizer_config.get('scheduler', None) == 'linear':
            from transformers import get_linear_schedule_with_warmup
            # Using Linear scheduler
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
                last_epoch=self.global_step,
            )
        elif self.optimizer_config.get('scheduler', None) == 'cosine':
            t_total = self.training_config.trainer_paras.max_epochs
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                            t_total, eta_min=1e-5, last_epoch=-1, verbose=False)
        else:
            from transformers import get_constant_schedule_with_warmup
            # Using constant scheduler
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                last_epoch=self.global_step,
            )
        
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                # REQUIRED: The scheduler instance
                "scheduler": self.scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            }
        }

    # def train_dataloader(self):
    #     return DataLoader(
    #         self.train_dataset,
    #         shuffle=True,
    #         batch_size=self.training_config['batch_size'],
    #         num_workers=self.training_config.get('dataloader_workers', 8)
    #     )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_dataset,
    #         shuffle=False,
    #         batch_size=self.training_config['batch_size'],
    #         num_workers=self.training_config.get('dataloader_workers', 8)
    #     )
    
    # def test_dataloader(self):
    #     return DataLoader(
    #         self.test_dataset,
    #         shuffle=False,
    #         batch_size=self.test_config['batch_size'],
    #         num_workers=self.test_config.get('dataloader_workers', 8)
    #     )

    def train_dataloader(self):
        if 'data_loaders' in self.__dict__:
            self.train_dataloader_names = list(self.data_loaders['train'].keys())
        
            # TODO: we only allow one train data loader at the moment
            return self.train_dataloaders[0]
        elif 'train_dataset' in self.__dict__:
            return DataLoader(
                self.train_dataset,
                shuffle=True,
                batch_size=self.training_config['batch_size'],
                num_workers=self.training_config.get('dataloader_workers', 8)
            )
        else:
            raise NotImplementedError('Either data_loaders or train_dataset must be available before train_dataloader() is called')
    
    def val_dataloader(self):
        if 'data_loaders' in self.__dict__:
            self.val_dataloader_names = list(self.data_loaders['valid'].keys())
            return self.valid_dataloaders
        elif 'val_dataset' in self.__dict__:
            return DataLoader(
                self.val_dataset,
                shuffle=False,
                batch_size=self.training_config['batch_size'],
                num_workers=self.training_config.get('dataloader_workers', 8)
            ) 
        else:
            raise NotImplementedError('Either data_loaders or val_dataset must be available before val_dataloader() is called')

    
    def test_dataloader(self):
        if 'data_loaders' in self.__dict__:
            self.test_dataloader_names = list(self.data_loaders['test'].keys())
            return self.test_dataloaders
        elif 'test_dataset' in self.__dict__:
            return DataLoader(
                self.test_dataset,
                shuffle=False,
                batch_size=self.test_config['batch_size'],
                num_workers=self.test_config.get('dataloader_workers', 8)
            )
        else:
            raise NotImplementedError('Either data_loaders or test_dataset must be available before test_dataloader() is called')
            
            


    def on_exception(self, trainer, pl_module, exception):
        # handle exception
        if self.wandb_logger and trainer.is_global_zero:
            if self.wandb_logger.experiment is not None:
                logger.error(f"Attempting to stop the wandb run {self.wandb_logger.experiment}")
                self.wandb_logger.experiment.finish()
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def on_validation_start(self) -> None: 
        self.valid_cnt += 1 
        base_recorder_dir = self.global_config.get('train_dir', '/tmp')
        recorder_name = f"validation-{self.valid_cnt}-{self.global_step}" 
        self.valid_eval_recorder = EvalRecorder(recorder_name, base_recorder_dir, meta_config=copy.copy(self.global_config))
        self.valid_eval_recorder.meta_config.update({'valid_run_count': self.valid_cnt, 'global_step': self.global_step})

    def on_validation_end(self) -> None:
        self.valid_eval_recorder.save_to_disk(f"eval_recorder", file_format='json')
        print("Validation recorder saved to", self.valid_eval_recorder.save_dir)


    def on_test_start(self) -> None: 
        base_recorder_dir = self.global_config['experiment_name'], self.global_config.get('test_dir', '/tmp' )
        recorder_name = f"test-evaluation"
        self.test_eval_recorder = EvalRecorder(recorder_name, base_recorder_dir, meta_config=copy.copy(self.global_config))

    def on_test_end(self) -> None: 
        self.test_eval_recorder.save_to_disk(f"eval_recorder", file_format='json')
        print("Test evaluation recorder saved to", self.test_eval_recorder.save_dir)
    
    def on_train_end(self) -> None: 
        if 'valid_eval_recorder' in self.__dict__:
            self.valid_eval_recorder.rename(new_name='validation_last')
            self.valid_eval_recorder.save_to_disk("eval_recorder", file_format='json')
        print("Training completes!")
