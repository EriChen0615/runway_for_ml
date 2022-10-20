import os
import pickle
from easydict import EasyDict
import logging
logger = logging.getLogger(__name__)
from utils.dirs import create_dirs
from typing import Dict


def save_cached_data(config, data_to_save, data_name, data_path=''):

    # Create the folder if not exist
    if not os.path.exists(config.cache.default_folder):
        create_dirs([config.cache.default_folder])

    if not data_path:
        data_path = os.path.join(
            config.cache.default_folder,
            '{}.pkl'.format(data_name))
    else:
        data_path = os.path.join(data_path, '{}.pkl'.format(data_name))

    with open(data_path, "wb" ) as f:
        logger.info('saving preprocessed data...')
        dump_data = {
            'cache': data_to_save,
        }
        pickle.dump(dump_data, f)
        logger.info(f'preprocessed data has been saved to {data_path}')



def load_cached_data(config, data_name, data_path='', condition=True):
    """[summary]

    Args:
        config ([type]): [description]
        data_name ([type]): [description]
        data_path (str, optional): [description]. Defaults to ''.
        condition (bool, optional): some condition to decide whether to load the cache. Defaults to True.

    Returns:
        [type]: [description]
    """
    if not data_path:
        data_path = os.path.join(
            config.cache.default_folder,
            '{}.pkl'.format(data_name))
    else:
        data_path = os.path.join(data_path, '{}.pkl'.format(data_name))
    
    # Check if config indicates to regenerate
    if config.cache.regenerate[data_name]:
        logger.info('Data "{}" is forced to be re-generated by config file.'.format(data_name))
        return None

    if os.path.exists(data_path) and condition:
        try:
            # Read data instead of re-generate
            logger.info(f'reading preprocessed data from {data_path}')
            with open(data_path, "rb" ) as f:
                load_pickle_data = pickle.load(f)['cache']
                return EasyDict(load_pickle_data)
        except Exception as e:
            logger.error('Failed to load pre-processed data, skipping...')
            logger.error(str(e))
    else:
        # This data is not cached
        return None

def cache_file_exists(data_file_name):
    return os.path.exists(data_file_name)
    
def cache_data_to_disk(
    data_to_save: Dict[str, any],
    data_name: str,
    dir_path: str, 
    save_format: str = 'pkl',
    ):
    """
    cache data_to_dist to disk at location `dir_path/data_name`
    """
    if not os.path.exists(dir_path):
        create_dirs([dir_path])
    
    data_file_name = os.path.join(dir_path, f"{data_name}.{save_format}")
    
    if save_format == 'pkl':
        save_pickle_data(data_to_save, data_file_name)
        print(f"Data saved to {data_file_name}")
    else:
        raise NotImplementedError(f"Saving data to disk with {save_format} is not implemented!")

def make_cache_file_name(
    data_name: str,
    dir_path: str,
    save_format: str = 'pkl',
    ):
    return os.path.join(dir_path, f"{data_name}.{save_format}")


def load_data_from_disk(
    data_name: str,
    dir_path: str,
    save_format: str = 'pkl',
    ):
    data_file_name = make_cache_file_name(data_name, dir_path=dir_path, save_format=save_format)
    if os.path.exists(data_file_name):
        if save_format == 'pkl':
            loaded_data = load_pickle_data(data_file_name)
            print(f"Data loaded from {data_file_name}")
            return loaded_data
        else:
            raise NotImplementedError(f".{save_format} loading is not implemented in cache system!")
    else:
        return None # data doesn't exist
        

def save_pickle_data(data_to_save, data_file_name):
    with open(data_file_name, 'wb') as f:
        dump_data = {
            'cache': data_to_save,
        }
        pickle.dump(dump_data, f)

def load_pickle_data(data_file_name):
    with open(data_file_name, 'rb') as f:
        load_pickle_data = pickle.load(f)['cache']
    return EasyDict(load_pickle_data)







    