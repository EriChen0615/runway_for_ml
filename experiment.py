"""
Experiment System:
    - Manage experiment hierarchy, versioning, etc.
"""
import os
from pathlib import Path
from .utils.global_variables import Executor_Registry, DataTransform_Registry
from .utils import config_system as rw_conf
from .utils import util 
from .configs import configuration as rw_cfg
from .data_module.data_pipeline import DataPipeline 


import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from easydict import EasyDict
import json
import pandas as pd

class RunwayExperiment:
    def __init__(self, config_dict, root_dir=None):
        self.config_dict = config_dict
        self.exp_name = config_dict.get('experiment_name', None)
        self.tag = config_dict.get('tag', None)
        self.test_suffix = config_dict.get('test_suffix')
        self.meta_conf = self.config_dict['meta']

        self.root_exp_dir = root_dir or Path(self.meta_conf['EXPERIMENT_FOLDER'])
        self.exp_full_name = None
        self.exp_dir = None

        self.rw_executor = None
    
    def _make_exp_full_name(self, exp_name, ver_num, tag):
        if tag is None:
            return f"{exp_name}_V{ver_num}"
        else:
            return f"{exp_name}_V{ver_num}_tag:{tag}"

    def _make_experiment_dir(self, root_exp_dir, exp_name, ver_num, tag):
        self.exp_full_name = self._make_exp_full_name(exp_name, ver_num, tag)
        return root_exp_dir / self.exp_full_name 
    
    def _check_version_and_update_exp_dir(self):
        while os.path.exists(self.exp_dir):
            self.ver_num += 1
            self.exp_dir = self._make_experiment_dir(self.root_exp_dir, self.exp_name, self.ver_num, self.tag)

    def init_loggers(self, mode='train'):
        self.logger_enable = self.meta_conf['logger_enable']
        print("Using loggers:", self.logger_enable)
        loggers = []
        if mode == 'train':
            log_dir = self.train_log_dir
        elif mode == 'test':
            log_dir = self.test_dir
        for logger_type in self.logger_enable:
            if logger_type == "csv":
                csv_logger = pl_loggers.CSVLogger(save_dir=log_dir)
                loggers.append(csv_logger)
            elif logger_type == "tensorboard":
                tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, sub_dir='tb_log')
                loggers.append(tb_logger)
            elif logger_type == 'wandb':
                assert "WANDB" in self.meta_conf, "WANDB configuration missing in config file, but wandb_logger is used"
                wandb_conf = self.meta_conf["WANDB"]
                wandb_logger = pl_loggers.WandbLogger(
                    name=self.exp_full_name,
                    project=wandb_conf['PROJECT_NAME'],
                )
                loggers.append(wandb_logger)
        return loggers
    
    def init_executor(self, mode='train'):
        meta_config = self.config_dict.meta
        dp_config = self.config_dict.data_pipeline
        executor_config = self.config_dict.executor
        train_config = self.config_dict.train
        test_config = self.config_dict.test

        loggers = self.init_loggers(mode=mode) # initialize loggers
        print(loggers)

        tokenizer = util.get_tokenizer(self.config_dict.tokenizer_config)

        rw_executor = None
        if mode == 'train':
            rw_executor = Executor_Registry[executor_config.ExecutorClass](
                data_pipeline_config=dp_config,
                model_config=executor_config.model_config,
                mode='train',
                train_config=train_config,
                logger=loggers,
                tokenizer=tokenizer,
                **executor_config.init_kwargs
            )
        elif mode == 'test':
            load_ckpt_path = self.train_dir / "lightning_logs" / "version_0" / "checkpoints" / test_config['checkpoint_name']
            log_file_path = self.test_dir / 'test_case.txt'
            print("Loading checkpoint at:", load_ckpt_path)
            print("Saving testing results to:", log_file_path)
            rw_executor = Executor_Registry[executor_config.ExecutorClass].load_from_checkpoint(
                load_ckpt_path,
                data_pipeline_config=dp_config,
                model_config=executor_config.model_config,
                mode='test',
                test_config=test_config,
                logger=loggers,
                log_file_path=log_file_path,
                tokenizer=tokenizer,
                **executor_config.init_kwargs
            )
        return rw_executor
    
    def save_config_to(self, dir_path):
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        file_path = dir_path / 'config.json'
        with open(file_path, 'w') as f:
            json.dump(self.config_dict, f)

    def train(self):
        train_config = self.config_dict.train

        self.ver_num = 0
        self.exp_dir = self._make_experiment_dir(self.root_exp_dir, self.exp_name, self.ver_num, self.tag)
        self._check_version_and_update_exp_dir()
        self.config_dict['exp_version'] = self.ver_num

        self.train_dir = self.exp_dir / 'train'
        self.train_log_dir = self.train_dir / 'logs'

        self.rw_executor = self.init_executor(mode='train') 
        checkpoint_callback = ModelCheckpoint(**train_config.model_checkpoint_callback_paras)

        self.save_config_to(self.train_dir)

        trainer = pl.Trainer(**train_config.get('trainer_paras', {}), default_root_dir=self.train_dir ,callbacks=[checkpoint_callback])
        trainer.fit(self.rw_executor)
    
    def test(self):
        test_config = self.config_dict.test


        assert 'exp_version' in self.config_dict, "You need to specify experiment version to run test!"
        self.ver_num = self.config_dict['exp_version']
        self.exp_dir = self._make_experiment_dir(self.root_exp_dir, self.exp_name, self.ver_num, self.tag)
        self.train_dir = self.exp_dir / 'train'
        self.test_dir = self.exp_dir / f'test-{self.test_suffix}'

        print('test-directory:', self.test_dir)
        self.save_config_to(self.test_dir)

        self.rw_executor = self.init_executor(mode='test')
        trainer = pl.Trainer(**test_config.get('trainer_paras', {}), default_root_dir=self.test_dir)
        trainer.test(self.rw_executor)
    
    def eval(self):
        eval_config = self.config_dict.eval
        assert 'exp_version' in self.config_dict, "You must experiment version to evaluate"
        assert 'test_suffix' in self.config_dict, "You must specify name of the test run"
        assert 'eval_op_name' in eval_config, "You must specify name of the evaluation op in .eval"
        
        self.ver_num = self.config_dict['exp_version']
        self.exp_dir = self._make_experiment_dir(self.root_exp_dir, self.exp_name, self.ver_num, self.tag)
        self.test_dir = self.exp_dir / f'test-{self.test_suffix}'

        eval_op_name = eval_config['eval_op_name']
        eval_op_kwargs = eval_config.get('setup_kwargs', {})
        eval_op = DataTransform_Registry[eval_op_name]()
        eval_op.setup(**eval_op_kwargs)

        test_df = pd.read_csv(self.test_dir / 'test_case.csv')
        eval_res_dict = eval_op._call(test_df)

        metric_df = eval_res_dict['metrics']
        anno_df = eval_res_dict['annotations']

        metric_df.to_csv(self.test_dir / 'metrics.csv')
        anno_df.to_csv(self.test_dir / 'annotated_test_case.csv')

        print("Saved to:", self.test_dir)
        print("Evaluation completes!")

