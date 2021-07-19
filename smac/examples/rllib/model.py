from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ray.rllib.models import Model
from ray.rllib.models.tf.misc import normc_initializer


class MaskedActionsModel(Model):
    """Custom RLlib model that emits -inf logits for invalid actions.

    This is used to handle the variable-length StarCraft action space.
    """

    def _build_layers_v2(self, input_dict, num_outputs, options):
        action_mask = input_dict["obs"]["action_mask"]
        if num_outputs != action_mask.shape[1].value:
            raise ValueError(
                "This model assumes num outputs is equal to max avail actions",
                num_outputs,
                action_mask,
            )

        # Standard fully connected network
        last_layer = input_dict["obs"]["obs"]
        hiddens = options.get("fcnet_hiddens")
        for i, size in enumerate(hiddens):
            label = "fc{}".format(i)
            last_layer = tf.layers.dense(
                last_layer,
                size,
                kernel_initializer=normc_initializer(1.0),
                activation=tf.nn.tanh,
                name=label,
            )
        action_logits = tf.layers.dense(
            last_layer,
            num_outputs,
            kernel_initializer=normc_initializer(0.01),
            activation=None,
            name="fc_out",
        )

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        masked_logits = inf_mask + action_logits

        return masked_logits, last_layer
