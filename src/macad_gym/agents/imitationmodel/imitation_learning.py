from __future__ import print_function

import os

import tensorflow as tf
import numpy as np
import tf_slim as slim
import torch
from macad_gym.agents.imitationmodel.imitation_learning_network import load_imitation_learning_network

class ImitationLearning:
    def __init__(self, sess, memory_fraction=0.25, image_cut=(115, 510)):
        self.dropout_vec = [1.0] * 8 + [0.7] * 2

        self._image_size = (88, 200, 3)
        self._sess = sess

        self._input_images = tf.placeholder("float", shape=[None, self._image_size[0],
                                                            self._image_size[1],
                                                            self._image_size[2]],
                                            name="input_image")

        self._dout = tf.placeholder("float", shape=[len(self.dropout_vec)])

        with tf.name_scope("Network"):
            self._network_tensor = load_imitation_learning_network(self._input_images,
                                                                   self._image_size, self._dout)

        import os
        dir_path = os.path.dirname(__file__)

        self._models_path = dir_path + '/model/'

        self._image_cut = image_cut
        self.variables_to_restore = tf.global_variables()

    def load_model(self):

        saver = tf.train.Saver(self.variables_to_restore, max_to_keep=0)

        if not os.path.exists(self._models_path):
            raise RuntimeError('failed to find the models path')

        ckpt = tf.train.get_checkpoint_state(self._models_path)
        if ckpt:
            print('Restoring from ', ckpt.model_checkpoint_path)
            saver.restore(self._sess, ckpt.model_checkpoint_path)
        else:
            ckpt = 0

        return ckpt

    def load_finetuned_encoder_weights(self, checkpoint_path):
        """Copy fine-tuned PyTorch image weights into the legacy TF graph.

        The downstream policy continues to call ``compute_feature`` on this
        original TensorFlow encoder. PyTorch BatchNorm's learned scale is
        folded into each convolution for compatibility with the legacy TF
        batch_norm layers, which were created with ``scale=False``.
        """
        checkpoint = torch.load(
            os.path.abspath(os.path.expanduser(checkpoint_path)),
            map_location="cpu",
        )
        state = checkpoint.get("image_encoder")
        if not isinstance(state, dict):
            raise ValueError(
                "Fine-tuned checkpoint has no image_encoder state: {}".format(
                    checkpoint_path
                )
            )

        variables = self._sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        def find_variable(name):
            matches = [
                variable for variable in variables
                if variable.name.split(":", 1)[0] == name
                or variable.name.split(":", 1)[0].endswith("/" + name)
            ]
            if len(matches) != 1:
                raise KeyError(
                    "Expected one legacy TF variable ending in {!r}, found {}".format(
                        name, len(matches)
                    )
                )
            return matches[0]

        assignments = []

        def assign(name, value):
            value = np.asarray(value, dtype=np.float32)
            variable = find_variable(name)
            expected_shape = tuple(variable.shape.as_list())
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    "Fine-tuned image weight {} has shape {}, expected {}".format(
                        name, tuple(value.shape), expected_shape
                    )
                )
            assignments.append(tf.assign(variable, value))

        def as_numpy(name):
            if name not in state:
                raise KeyError("Missing fine-tuned image parameter: {}".format(name))
            return state[name].detach().cpu().numpy()

        epsilon = 1.0e-3
        for index in range(1, 9):
            block = index - 1
            prefix = "backbone.{}.block.".format(block)
            conv_weight = as_numpy(prefix + "0.weight")
            conv_bias = as_numpy(prefix + "0.bias")
            bn_prefix = prefix + "1."
            gamma = as_numpy(bn_prefix + "weight")
            beta = as_numpy(bn_prefix + "bias")
            moving_mean = as_numpy(bn_prefix + "running_mean")
            moving_variance = as_numpy(bn_prefix + "running_var")

            scale = gamma / np.sqrt(moving_variance + epsilon)
            # PyTorch OIHW -> TensorFlow HWIO, then fold inference BN affine
            # parameters into the convolution so the original TF graph stays
            # unchanged and runs the same inference path.
            tf_kernel = conv_weight.transpose(2, 3, 1, 0)
            tf_kernel = tf_kernel * scale.reshape((1, 1, 1, -1))
            tf_bias = (conv_bias - moving_mean) * scale + beta
            assign("W_c_{}".format(index), tf_kernel)
            # TF1 exposes these bias variables as Variable under the name scope.
            assign("conv_block{}/Variable".format(block), tf_bias)

            # Make the legacy scale=False TF BatchNorm an identity.
            assign("bn{}/beta".format(index), np.zeros_like(beta))
            assign("bn{}/moving_mean".format(index), np.zeros_like(moving_mean))
            assign(
                "bn{}/moving_variance".format(index),
                np.full_like(moving_variance, 1.0 - epsilon),
            )

        for index, layer in enumerate(("fc1", "fc2"), 1):
            weight = as_numpy(layer + ".weight").transpose(1, 0)
            bias = as_numpy(layer + ".bias")
            assign("W_f_{}".format(index), weight)
            assign("fc{}/Variable".format(index), bias)

        self._sess.run(assignments)
        print(
            "Loaded fine-tuned image weights into the original TensorFlow "
            "encoder ({} tensors)".format(len(assignments))
        )

    def compute_feature(self, sensor_data):
        image_input = sensor_data
        # image_input = np.multiply(image_input, 1.0 / 255.0)
        image_input = image_input.reshape(
            (1, self._image_size[0], self._image_size[1], self._image_size[2]))
        print("[图像编码器] 预处理后输入: {}".format(image_input.shape))

        feedDict = {self._input_images: image_input, self._dout: [1] * len(self.dropout_vec)}

        output_vector = self._sess.run(self._network_tensor, feed_dict=feedDict)
        return output_vector[0]
