SMAC on RLlib
=============

This example shows how to run SMAC environments with RLlib multi-agent.

Instructions
------------

To get started, first install RLlib with ``pip install -U ray[rllib]``. You will also need TensorFlow installed.

In ``run_ppo.py``, each agent will be controlled by an independent PPO policy (the policies share weights). This setup serves as a single-agent baseline for this task.

In ``run_qmix.py``, the agents are controlled by the multi-agent QMIX policy. This setup is an example of centralized training and decentralized execution.

See https://ray.readthedocs.io/en/latest/rllib.html for more documentation.
