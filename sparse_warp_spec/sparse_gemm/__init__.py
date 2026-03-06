import os
import torch
import torch.utils.cpp_extension

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# Import functions from the CPP module
import sparse_gemm_cpp
sparse_gemm_cpp.init(
    os.path.dirname(os.path.abspath(__file__)), # Library root directory path
    torch.utils.cpp_extension.CUDA_HOME         # CUDA home
)

# Configs
from sparse_gemm_cpp import (
    set_num_sms,
    get_num_sms
)

# Kernels
from sparse_gemm_cpp import (
    ssd_naive,
    ssd,
    dss
)

# Some utils
from . import testing
from . import utils
from .utils import *
