from __future__ import print_function

import hashlib
import json
import os
import random

import numpy as np


def seed_everything(seed):
    """Seed RNGs used by the lane-change experiments."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        if hasattr(tf, "set_random_seed"):
            tf.set_random_seed(seed)
        elif hasattr(tf.random, "set_seed"):
            tf.random.set_seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return seed


def ensure_dir(path):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def file_sha256(path):
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
