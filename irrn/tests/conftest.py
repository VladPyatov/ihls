import gc
import os
import random
import subprocess as sp

import numpy as np
import pytest
import torch as th
from _pytest.config import Config
from _pytest.config.argparsing import Parser


def pytest_configure(config: Config) -> None:
    pytest.grads_autograd = None
    pytest.grads_custom = None


def pytest_addoption(parser: Parser) -> None:
    for i, elem in enumerate(parser._getparser()._actions):
        if elem.option_strings == ['--capture']:
            elem.default = 'no'
    group = parser.getgroup("xdist", "distributed and subprocess testing")
    group.options[0].default = 12


def enable_deterministic():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    th.use_deterministic_algorithms(True)
    random.seed(55)
    np.random.seed(55)
    th.manual_seed(55)


def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    cmd = "/usr/local/nvidia/bin/nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(cmd.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def set_gpu_with_max_memory():
    memo_avail = get_gpu_memory()
    gpu_id = np.argmax(memo_avail)
    th.cuda.set_device(f'cuda:{gpu_id}')


def pytest_runtest_call(item: "Item") -> None:
    """Called to run the test for test item (the call phase).

    The default implementation calls ``item.runtest()``.
    """
    set_gpu_with_max_memory()
    enable_deterministic()
    gc.collect()
    th.cuda.empty_cache()
