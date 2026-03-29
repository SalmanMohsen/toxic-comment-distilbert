"""
utils/seed.py
-------------
Centralised seed control for reproducible experiments.
"""

import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """
    Set seeds across every library that introduces randomness.

    Libraries seeded
    ----------------
    - Python stdlib  (random)
    - NumPy          (numpy.random)
    - PyTorch CPU    (torch.manual_seed)
    - PyTorch CUDA   (torch.cuda.manual_seed_all — covers multi-GPU)
    - OS hash seed   (PYTHONHASHSEED env var)

    Remaining nondeterminism
    ------------------------
    cuDNN uses non-deterministic atomic operations inside the scaled-dot-product
    attention kernel.  We set ``deterministic=True`` which forces the slower
    deterministic path, but a small residual variance may remain depending on
    the CUDA driver version.

    Parameters
    ----------
    seed : int
        The global seed value (e.g. 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Force deterministic CUDA ops — may slow training slightly
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
