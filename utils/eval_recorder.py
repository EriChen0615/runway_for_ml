import pandas as pd
import pickle
from collections import defaultdict
from easydict import EasyDict
import os
import wandb

class EvalRecorder:
    def __init__(
        self,
        name,
        base_dir,
        meta_config={},
    ):
        self.name = name
        self.base_dir = base_dir

        self.meta_config = meta_config

        self._sample_logs = defaultdict(list) 
        self._stats_logs = defaultdict(list)
    
    @property
    def save_dir(self):
        return os.path.join(self.base_dir, self.name)
        
    def _make_file_path(self, file_name):
        file_path = os.path.join(self.save_dir, file_name)
        return file_path

    @classmethod
    def load_from_disk(cls, file_prefix, file_format='pkl'): #TODO
        """load a saved recorder from disk

        :param file_prefix: _description_
        :param file_format: _description_
        """
        if file_format =='pkl':
            pass
        elif file_format == 'csv':
            pass
        pass
    
    def _convert_to_dataframe(self, dict, *args, **kwargs):
        df = pd.DataFrame(dict, *args, **kwargs)
        return df
        

    def save_to_disk(self, file_prefix, file_format='pkl'): #TODO
        """save the recorder to file system

        :param file_prefix: _description_
        :param file_format: _description_, defaults to 'pkl'
        """
        if file_format == 'pkl':
            pass
        elif file_format == 'csv':
            pass
        pass
    
    def log_sample_dict(self, sample_dict): #TODO
        """log a dictionary that corresponds to a sample level inference/evaluation results or metric

        :param sample_dict: _description_
        """
        pass

    def log_stats_dict(self, stats_dict): #TODO
        """log a dictionary that corresponds to a dataset level statistics

        :param stats_dict: _description_
        """
        pass

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
        return len(self._sample_logs)
    
    def __getitem__(self ,index):
        return {colname: column[index] for colname, column in self._sample_logs.items()}

    def upload_to_wandb(self, prefix='test', no_log_stats=[]):
        """_summary_

        :param prefix: _description_, defaults to 'test'
        """
        assert 'wandb_config' in self.meta_config, "to upload to wandb, wandb_config must be present in self.meta_config"
        sample_table = self._convert_to_dataframe(self._sample_logs)
        table_to_log = wandb.Table(data=sample_table.values.tolist(), columns=sample_table.columns.tolist())
        wandb.log({f"{prefix}/Sample Table": table_to_log})
        
        for stat_name, value in self._stats_logs.items():
            if stat_name in no_log_stats:
                continue
            wandb.log({f"{prefix}/{stat_name}": value})
