import os
import pickle
import json
import tempfile
import shutil
import copy
from collections import defaultdict
import pytest

from runway_for_ml.utils.eval_recorder import EvalRecorder

@pytest.fixture(scope="module")
def tempdir(request):
    # create a temporary directory for testing
    dirpath = "tests/eval_recorder_test"

    def remove_tempdir():
        # remove the temporary directory after testing
        shutil.rmtree(dirpath)

    request.addfinalizer(remove_tempdir)

    return dirpath

def test_init():
    # test the __init__ method of EvalRecorder
    recorder = EvalRecorder(name="test_recorder", base_dir="tests/tmp")
    assert recorder.name == "test_recorder"
    assert recorder.base_dir == "tests/tmp"
    assert recorder.meta_config == {"name": "test_recorder", "base_dir": "tests/tmp"}
    assert recorder._log_index == 0
    assert recorder._sample_logs == defaultdict(list)
    assert recorder._sample_columns == set()
    assert recorder._stats_logs == defaultdict(list)

def test_rename():
    # test the rename method of EvalRecorder
    recorder = EvalRecorder(name="test_recorder", base_dir="tests/tmp")
    recorder.rename("new_name")
    assert recorder.name == "new_name"
    assert recorder.meta_config == {"name": "new_name", "base_dir": "tests/tmp"}
    recorder.rename("new_name", "tests/new_dir")
    assert recorder.base_dir == "tests/new_dir"
    assert recorder.meta_config == {"name": "new_name", "base_dir": "tests/new_dir"}

def test_save_to_disk_and_load_from_disk(tempdir):
    # test the save_to_disk and load_from_disk methods of EvalRecorder
    recorder = EvalRecorder(name="test_recorder", base_dir=tempdir)
    recorder.save_to_disk("eval_recorder", file_format="pkl")
    assert os.path.exists(os.path.join(tempdir, "test_recorder", "eval_recorder.pkl"))
    loaded_recorder = EvalRecorder.load_from_disk("test_recorder", tempdir, "eval_recorder", file_format="pkl")
    assert loaded_recorder.name == "test_recorder"
    assert loaded_recorder.base_dir == tempdir

def test_reset_for_new_pass():
    # test the reset_for_new_pass method of EvalRecorder
    recorder = EvalRecorder(name="test_recorder", base_dir="/tmp")
    recorder.reset_for_new_pass()
    assert recorder._log_index == 0

def test_copy_data_from():
    # test the copy_data_from method of EvalRecorder
    recorder1 = EvalRecorder(name="test_recorder", base_dir="/tmp")
    recorder1._sample_logs["sample"] = [1, 2, 3]
    recorder1._stats_logs["stat"] = [4, 5, 6]
    recorder1.meta_config["config"] = {"foo": "bar"}
    recorder2 = EvalRecorder(name="new_recorder", base_dir="/tmp")
    recorder2.copy_data_from(recorder1)
    assert recorder2._sample_logs == recorder1._sample_logs
    assert recorder2._stats_logs == recorder1._stats_logs
    assert recorder2.meta_config == recorder1.meta_config
