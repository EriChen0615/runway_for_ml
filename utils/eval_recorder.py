import pandas as pd
import pickle
from collections import defaultdict
from easydict import EasyDict
import os
# import wandb
import json
import copy
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from pathlib import Path

class EvalRecorder:
    def __init__(
        self,
        name=None,
        base_dir=None,
        meta_config={},
    ):
        self.name = name
        self.base_dir = base_dir

        self.meta_config = meta_config
        self.meta_config['name'] = name
        self.meta_config['base_dir'] = base_dir

        self._log_index = 0
        self._sample_logs = defaultdict(list) 
        self._sample_logs['index'] = []
        self._stats_logs = defaultdict(list)
    
    def rename(self, new_name, new_base_dir=None):
        self.name = new_name
        self.meta_config['name'] = self.name
        if new_base_dir:
            self.base_dir = new_base_dir
            self.meta_config['base_dir'] = self.base_dir
    
    @property
    def save_dir(self):
        return os.path.join(self.base_dir, self.name)
        
    def _make_file_path(self, file_name, file_format):
        file_path = os.path.join(self.save_dir, f"{file_name}.{file_format}")
        return Path(file_path)
    
    def reset_for_new_pass(self):
        """reset for another new pass through the dataset
        """
        self._log_index = 0
    
    def _get_separate_serialize_filenames(self, file_prefix, file_format):
        sample_log_file_path = self._make_file_path(f"{file_prefix}-sample_log", file_format=file_format)
        stats_log_file_path = self._make_file_path(f"{file_prefix}-stats_log", file_format=file_format)
        meta_config_file_path = self._make_file_path(f"{file_prefix}-meta_config", file_format=file_format)
        return sample_log_file_path, stats_log_file_path, meta_config_file_path
        

    def save_to_disk(self, file_prefix, file_format='pkl'): 
        """save the recorder to file system

        :param file_prefix: _description_
        :param file_format: _description_, defaults to 'pkl'
        """
        if file_format == 'pkl':
            file_path = self._make_file_path(file_prefix, file_format=file_format)
            os.makedirs(file_path.parent, exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"{self.name} EvalRecorder saved to {file_path}")
        elif file_format == 'json':
            sample_log_file_path, stats_log_file_path, meta_config_file_path = self._get_separate_serialize_filenames(file_prefix, file_format)
            os.makedirs(sample_log_file_path.parent, exist_ok=True)
            with open(sample_log_file_path, 'w') as f:
                json.dump(self._sample_logs, f)
            with open(stats_log_file_path, 'w') as f:
                json.dump(self._stats_logs, f)
            with open(meta_config_file_path, 'w') as f:
                json.dump(self.meta_config, f)
            logger.info(f"{self.name} EvalRecorder saved to {sample_log_file_path}, {stats_log_file_path}, {meta_config_file_path}") 
        else:
            raise NotImplementedError()

    @classmethod
    def load_from_disk(cls, name, base_dir, file_prefix, file_format='pkl'): 
        """load a saved recorder from disk
        Before this is called. self.name and self.base_dir must be set correctly when the initial object is initialized

        :param file_prefix: _description_
        :param file_format: _description_
        :return: True if loading was successful, False otherwise.
        """
        instance = cls(name=name, base_dir=base_dir)
        if file_format =='pkl':
            file_path = instance._make_file_path(file_prefix, file_format=file_format)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    loaded_instance = pickle.load(f)
                    instance.copy_data_from(loaded_instance)
            else:
                return None
        elif file_format == 'json':
            sample_log_file_path, stats_log_file_path, meta_config_file_path = instance._get_separate_serialize_filenames(file_prefix, file_format)
            if os.path.exists(sample_log_file_path) and os.path.exists(stats_log_file_path) and os.path.exists(meta_config_file_path):
                with open(sample_log_file_path, 'r') as f:
                    instance._sample_logs = json.load(f)
                with open(stats_log_file_path, 'r') as f:
                    instance._stats_logs = json.load(f)
                with open(meta_config_file_path, 'r') as f:
                    instance.meta_config = json.load(f)
            else:
                return None
        else:
            raise NotImplementedError()
        instance.reset_for_new_pass()
        return instance
    
    
    def copy_data_from(self, other): #TODO
        """Get a (shallow) copy from another EvalRecorder
        name is preserved
        """
        self._sample_logs = copy.copy(other.get_sample_logs())
        self._stats_logs = copy.copy(other.get_stats_logs())
        self.meta_config = copy.copy(other.meta_config)
        
    
    def _convert_to_dataframe(self, dict, *args, **kwargs):
        """_summary_

        :param dict: _description_
        :return: _description_
        """
        df = pd.DataFrame(dict, *args, **kwargs)
        return df
        
    def _append_to_sample_logs_col(self, colname, value, idx=None):
        if idx > len(self):
            raise RuntimeError(f"idx cannot be larger than sample_logs length: idx={idx}, length={len(self)}")
        if idx == len(self):
            self._sample_logs['index'].append(idx)
        # ensure idx <= len(self)
        if colname not in self._sample_logs: # make new column if necessary
            self._sample_logs[colname] = [None] * (len(self)-1)

        if idx > len(self._sample_logs[colname]):
            raise RuntimeError(f"impossible case in eval recorder. idx={idx}, len={len(self._sample_logs[colname])}")
        elif idx == len(self._sample_logs[colname]):
            self._sample_logs[colname].append(value)
        else:
            self._sample_logs[colname][idx] = value
    
    def log_sample_dict(self, sample_dict): 
        """log a dictionary that corresponds to a sample level inference/evaluation results or metric
        Note that the behavior depends on a STATEFUL counter self._log_index

        :param sample_dict: _description_
        """
        for k, v in sample_dict.items():
            self._append_to_sample_logs_col(colname=k, value=v, idx=self._log_index)
        if self._log_index == len(self) - 1:
            no_value_columns = set(self._sample_logs.keys()) - set(sample_dict.keys()) - set({'index'})
            for col in no_value_columns:
                if self._log_index == len(self._sample_logs[col]):
                    self._append_to_sample_logs_col(colname=col, value=None, idx=self._log_index)
        self._log_index += 1 

    def log_stats_dict(self, stats_dict): 
        """log a dictionary that corresponds to a dataset level statistics

        :param stats_dict: _description_
        """
        self._stats_logs.update(stats_dict)

    def get_sample_logs(self, data_format='dict'):
        """_summary_

        :param data_format: _description_, defaults to 'dict'
        :raises NotImplementedError: _description_
        :return: _description_
        """
        if data_format == 'dict':
            return self._sample_logs
        elif data_format == 'csv':
            self._convert_to_csv(self._sample_logs)
        else:
            raise NotImplementedError(f'data_format {data_format} not supported!')
    
    def get_stats_logs(self, data_format='dict'):
        """_summary_

        :param data_format: _description_, defaults to 'dict'
        :raises NotImplementedError: _description_
        :return: _description_
        """
        if data_format == 'dict':
            return self._stats_logs
        elif data_format == 'csv':
            self._convert_to_csv(self._stats_logs)
        else:
            raise NotImplementedError(f'data_format {data_format} not supported!')
    
    def merge(self, others):
        """merge with another EvalRecorder; append non-overlapping fields to sample dict, extend stats dict and meta dict

        :param others: _description_
        """
        for other in others:
            assert len(other) == len(self), "Error! Only EvalRecorder with the same number of rows can be merged!"
            # sample-level merge
            other_sample_logs = other.get_sample_logs()
            for other_key in other_sample_logs:
                if other_key not in self._sample_logs.keys():
                    self._sample_logs[other_key] = other_sample_logs[other]
            
            # dataset stats merge
            self._stats_logs.update(other.get_stats_logs())
        return self
    
    def __len__(self):
        """assume invariant: all columns in the sample_logs dict has the same length

        :return: _description_
        """
        return len(self._sample_logs['index'])
    
    def __getitem__(self ,index):
        return {colname: column[index] for colname, column in self._sample_logs.items()}
    
    # def __setattr__(self, col_name, col_value) -> None:
    #     assert len(col_value) == len(self), f"Length mismatch: {col_name}: {len(col_value)} versus {len(self)}. Only column of the same number of rows can be added to sample logs!"
    #     self._sample_logs[col_name] = col_value

    def set_sample_logs_column(self, col_name, col_value):
        assert len(col_value) == len(self), f"Length mismatch: {col_name}: {len(col_value)} versus {len(self)}. Only column of the same number of rows can be added to sample logs!"
        self._sample_logs[col_name] = col_value
    
    def set_sample_logs_data(self, data):
        """Directly set the sample logs. Move log_index to tail 

        :param data: _description_
        """
        col_length = None
        for column in data:
            col_length = col_length or len(data[column])
            if len(data[column]) != col_length:
                raise RuntimeError(f"All columns in the data used to set sample logs must have the same length. Unmatched column: {column}, length:{len(data[column])}")
        self._sample_logs = data
        self._sample_logs['index'] = [i for i in range(col_length)]
        self._log_index = len(self._sample_logs['index'])
    
    # def __getattr__(self, col_name):
    #     return self._sample_logs[col_name]

    def get_sample_logs_column(self, col_name):
        return self._sample_logs[col_name]

    # def upload_to_wandb(self, prefix='test', no_log_stats=[]):
    #     """_summary_

    #     :param prefix: _description_, defaults to 'test'
    #     """
    #     assert 'wandb_config' in self.meta_config, "to upload to wandb, wandb_config must be present in self.meta_config"
    #     sample_table = self._convert_to_dataframe(self._sample_logs)
    #     table_to_log = wandb.Table(data=sample_table.values.tolist(), columns=sample_table.columns.tolist())
    #     wandb.log({f"{prefix}/Sample Table": table_to_log})
        
    #     for stat_name, value in self._stats_logs.items():
    #         if stat_name in no_log_stats:
    #             continue
    #         wandb.log({f"{prefix}/{stat_name}": value})
