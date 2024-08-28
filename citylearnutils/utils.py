import os
import torch
import string
import random
import tempfile
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_dir(dir_name:str) -> str:
    """Create a directory if it doesn"t already exist"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def rand_str(n):
    """Creates a random alphanumeric string of length n"""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def get_temp_file(file_ext="tmp"):
    temp_name = next(tempfile._get_candidate_names())
    return f"/tmp/{temp_name}.{file_ext}"

