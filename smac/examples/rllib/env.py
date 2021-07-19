from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np

from gym.spaces import Discrete, Box, Dict

from ray import rllib

from smac.env import StarCraft2Env


class RLlibStarCraft2Env(rllib.MultiAgentEnv):
    """Wraps a smac StarCraft env to be compatible with RLlib multi-agent."""

    def __init__(self, **smac_args):
        """Create a new multi-agent StarCraft env compatible with RLlib.

        Arguments:
            smac_args (dict): Arguments to pass to the underlying
                smac.env.starcraft.StarCraft2Env instance.

        Examples:
            >>> from smac.examples.rllib import RLlibStarCraft2Env
            >>> env = RLlibStarCraft2Env(map_name="8m")
            >>> print(env.reset())
        """

        self._env = StarCraft2Env(**smac_args)
        self._ready_agents = []
        self.observation_space = Dict(
            {
                "obs": Box(-1, 1, shape=(self._env.get_obs_size(),)),
                "action_mask": Box(
                    0, 1, shape=(self._env.get_total_actions(),)
                ),
            }
        )
        self.action_space = Discrete(self._env.get_total_actions())

    def reset(self):
        """Resets the env and returns observations from ready agents.

        Returns:
            obs (dict): New observations for each ready agent.
        """

        obs_list, state_list = self._env.reset()
        return_obs = {}
        for i, obs in enumerate(obs_list):
            return_obs[i] = {
                "action_mask": np.array(self._env.get_avail_agent_actions(i)),
                "obs": obs,
            }

        self._ready_agents = list(range(len(obs_list)))
        return return_obs

    def step(self, action_dict):
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """

        actions = []
        for i in self._ready_agents:
            if i not in action_dict:
                raise ValueError(
                    "You must supply an action for agent: {}".format(i)
                )
            actions.append(action_dict[i])

        if len(actions) != len(self._ready_agents):
            raise ValueError(
                "Unexpected number of actions: {}".format(
                    action_dict,
                )
            )

        rew, done, info = self._env.step(actions)
        obs_list = self._env.get_obs()
        return_obs = {}
        for i, obs in enumerate(obs_list):
            return_obs[i] = {
                "action_mask": self._env.get_avail_agent_actions(i),
                "obs": obs,
            }
        rews = {i: rew / len(obs_list) for i in range(len(obs_list))}
        dones = {i: done for i in range(len(obs_list))}
        dones["__all__"] = done
        infos = {i: info for i in range(len(obs_list))}

        self._ready_agents = list(range(len(obs_list)))
        return return_obs, rews, dones, infos

    def close(self):
        """Close the environment"""
        self._env.close()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
