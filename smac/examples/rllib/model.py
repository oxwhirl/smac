from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from ray.rllib.models import Model
from ray.rllib.models.misc import normc_initializer


class MaskedActionsModel(Model):
    """Custom RLlib model that emits -inf logits for invalid actions.
    
    This is used to handle the variable-length StarCraft action space.
    """

    def _build_layers_v2(self, input_dict, num_outputs, options):
        action_mask = input_dict["obs"]["action_mask"]
        if num_outputs != action_mask.shape[1].value:
            raise ValueError(
                "This model assumes num outputs is equal to max avail actions",
                num_outputs, action_mask)

        # Standard fully connected network
        last_layer = input_dict["obs"]["obs"]
        hiddens = options.get("fcnet_hiddens")
        for i, size in enumerate(hiddens):
            label = "fc{}".format(i)
            last_layer = slim.fully_connected(
                last_layer,
                size,
                weights_initializer=normc_initializer(1.0),
                activation_fn=tf.nn.tanh,
                scope=label)
        action_logits = slim.fully_connected(
            last_layer,
            num_outputs,
            weights_initializer=normc_initializer(0.01),
            activation_fn=None,
            scope="fc_out")

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        masked_logits = inf_mask + action_logits

        return masked_logits, last_layer
