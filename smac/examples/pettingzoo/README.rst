SMAC on PettingZoo
==================

This example shows how to run SMAC environments with PettingZoo multi-agent API.

Instructions
------------

To get started, first install PettingZoo with ``pip install pettingzoo``.

The SMAC environment for PettingZoo, ``StarCraft2PZEnv``, can be initialized with two different API templates.
    * **AEC**: PettingZoo is based in the *Agent Environment Cycle* game model, more information about "AEC" can be read in the following `paper <https://arxiv.org/abs/2009.13051>`_. To create a SMAC environment as an "AEC" PettingZoo game model use: ::
        
        from smac.env.pettingzoo import StarCraft2PZEnv
        
        env = StarCraft2PZEnv.env()
    
    * **Parallel**: PettingZoo also supports parallel environments where all agents have simultaneous actions and observations. This type of environment can be created as follows: ::
        
        from smac.env.pettingzoo import StarCraft2PZEnv
        
        env = StarCraft2PZEnv.parallel_env()

`pettingzoo_demo.py` has an example of a SMAC environment being used as a PettingZoo "AEC" environment. With `env.render()` it is possible to output an instance of the environment as a frame in pygame. This is useful for debugging purposes.

| See https://www.pettingzoo.ml/api for more documentation.
