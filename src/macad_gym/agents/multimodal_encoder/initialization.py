"""Load the legacy HRAMA encoder weights for CARLA-domain fine-tuning."""

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import torch


def _checkpoint_prefix(path):
    path = os.path.abspath(os.path.expanduser(path))
    for suffix in (".index", ".meta", ".data-00000-of-00001"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
    return path


def _find_tensor(reader, candidates):
    shapes = reader.get_variable_to_shape_map()
    for candidate in candidates:
        exact = [name for name in shapes if name == candidate]
        suffix = [name for name in shapes if name.endswith("/" + candidate)]
        matches = exact or suffix
        if len(matches) == 1:
            return reader.get_tensor(matches[0]), matches[0]
    raise KeyError(
        "Could not find any of {} in TensorFlow checkpoint".format(candidates)
    )


def _copy_parameter(parameter, array, transpose=None, source_name=""):
    array = np.asarray(array)
    if transpose is not None:
        array = array.transpose(*transpose)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).to(
        dtype=parameter.dtype
    )
    if tuple(tensor.shape) != tuple(parameter.shape):
        raise ValueError(
            "Shape mismatch for {}: checkpoint {} vs model {}".format(
                source_name, tuple(tensor.shape), tuple(parameter.shape)
            )
        )
    with torch.no_grad():
        parameter.copy_(tensor)


def load_legacy_tensorflow_image_encoder(encoder, checkpoint_path):
    """Convert the original TF1 image checkpoint directly into the PyTorch model."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "Loading the legacy image checkpoint requires TensorFlow. Run this "
            "once in the existing HRAMA environment, or provide a converted "
            "PyTorch encoder checkpoint."
        ) from exc

    prefix = _checkpoint_prefix(checkpoint_path)
    if not os.path.isfile(prefix + ".index"):
        raise IOError("TensorFlow checkpoint not found: {}".format(prefix))
    reader = tf.train.load_checkpoint(prefix)
    loaded = []

    for index, block in enumerate(encoder.backbone, 1):
        conv = block.block[0]
        bn = block.block[1]
        weight, name = _find_tensor(reader, ["W_c_{}".format(index)])
        _copy_parameter(conv.weight, weight, (3, 2, 0, 1), name)
        loaded.append(name)
        bias, name = _find_tensor(
            reader,
            [
                "B_c_{}".format(index),
                "Network/conv_block{}/Variable".format(index - 1),
            ],
        )
        _copy_parameter(conv.bias, bias, source_name=name)
        loaded.append(name)

        for tf_name, torch_value in (
            ("beta", bn.bias),
            ("moving_mean", bn.running_mean),
            ("moving_variance", bn.running_var),
        ):
            value, name = _find_tensor(
                reader,
                [
                    "bn{}/{}".format(index, tf_name),
                    "batch_norm{}/{}".format(index, tf_name),
                ],
            )
            _copy_parameter(torch_value, value, source_name=name)
            loaded.append(name)
        # tf.contrib.layers.batch_norm used by the legacy model defaults to
        # scale=False, so most checkpoints have no gamma variable. PyTorch's
        # BatchNorm scale therefore remains at its equivalent default of one.
        try:
            gamma, name = _find_tensor(
                reader,
                [
                    "bn{}/gamma".format(index),
                    "batch_norm{}/gamma".format(index),
                ],
            )
        except KeyError:
            with torch.no_grad():
                bn.weight.fill_(1.0)
        else:
            _copy_parameter(bn.weight, gamma, source_name=name)
            loaded.append(name)

    for index, layer in enumerate((encoder.fc1, encoder.fc2), 1):
        weight, name = _find_tensor(reader, ["W_f_{}".format(index)])
        _copy_parameter(layer.weight, weight, (1, 0), name)
        loaded.append(name)
        bias, name = _find_tensor(
            reader,
            [
                "B_f_{}".format(index),
                "Network/fc{}/Variable".format(index),
            ],
        )
        _copy_parameter(layer.bias, bias, source_name=name)
        loaded.append(name)
    return loaded


def _unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a dictionary checkpoint")
    for key in ("image_encoder", "lidar_encoder", "model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def load_pytorch_encoder(encoder, checkpoint_path, prefix=""):
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        raise IOError("PyTorch checkpoint not found: {}".format(checkpoint_path))
    state = _unwrap_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    current = encoder.state_dict()
    matched = {}
    skipped = []
    for name, value in state.items():
        candidate = name
        if prefix and candidate.startswith(prefix):
            candidate = candidate[len(prefix):]
        if candidate.startswith("module."):
            candidate = candidate[len("module."):]
        if candidate in current and tuple(value.shape) == tuple(current[candidate].shape):
            matched[candidate] = value
        else:
            skipped.append(name)
    if not matched:
        raise ValueError(
            "No compatible encoder parameters found in {}".format(checkpoint_path)
        )
    current.update(matched)
    encoder.load_state_dict(current)
    return sorted(matched), sorted(skipped)


def load_image_encoder_initialization(encoder, checkpoint_path):
    prefix = _checkpoint_prefix(checkpoint_path)
    if os.path.isfile(prefix + ".index"):
        loaded = load_legacy_tensorflow_image_encoder(encoder, prefix)
        return "tensorflow", loaded, []
    loaded, skipped = load_pytorch_encoder(encoder, checkpoint_path)
    return "pytorch", loaded, skipped


def load_lidar_encoder_initialization(encoder, checkpoint_path):
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        raise IOError("PyTorch checkpoint not found: {}".format(checkpoint_path))
    state = _unwrap_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    current = encoder.state_dict()
    matched = {}
    skipped = []
    xyz_only_keys = {
        "sa1.conv_blocks.0.0.weight",
        "sa1.conv_blocks.1.0.weight",
        "sa1.conv_blocks.2.0.weight",
    }
    for name, value in state.items():
        candidate = name[len("module."):] if name.startswith("module.") else name
        if candidate not in current:
            skipped.append(name)
            continue
        target = current[candidate]
        if tuple(value.shape) == tuple(target.shape):
            matched[candidate] = value
            continue
        # The original ModelNet checkpoint was trained with XYZ + normals.
        # The CARLA LiDAR input contains XYZ only, exactly as handled by the
        # project's previous inference code: retain the XYZ channels and drop
        # the normal-vector channels in the first set-abstraction layer.
        if (
            candidate in xyz_only_keys
            and value.ndim == target.ndim
            and value.shape[0] == target.shape[0]
            and value.shape[1] >= target.shape[1]
            and tuple(value.shape[2:]) == tuple(target.shape[2:])
        ):
            matched[candidate] = value[:, : target.shape[1], ...]
        else:
            skipped.append(name)
    if not matched:
        raise ValueError(
            "No compatible PointNet++ parameters found in {}".format(
                checkpoint_path
            )
        )
    current.update(matched)
    encoder.load_state_dict(current)
    return sorted(matched), sorted(skipped)
