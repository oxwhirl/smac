from smac.env import StarCraft2Env
from smac.env.starcraft2.maps import get_map_params
from gymnasium.utils import EzPickle
from gymnasium.utils import seeding
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.conversions import (
    parallel_to_aec as from_parallel_wrapper,
)
from pettingzoo.utils import wrappers
import numpy as np


def parallel_env(max_cycles=None, **smac_args):
    if max_cycles is None:
        map_name = smac_args.get("map_name", "8m")
        max_cycles = get_map_params(map_name)["limit"]
    return _parallel_env(max_cycles, **smac_args)


def raw_env(max_cycles=None, **smac_args):
    return from_parallel_wrapper(parallel_env(max_cycles, **smac_args))


def make_env(raw_env):
    def env_fn(**kwargs):
        env = raw_env(**kwargs)
        # env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
        env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env_fn


class smac_parallel_env(ParallelEnv):
    def __init__(self, env, max_cycles):
        self.max_cycles = max_cycles
        self.env = env
        self.env.reset()
        self.reset_flag = 0
        self.agents, self.action_spaces = self._init_agents()
        self.possible_agents = self.agents[:]

        observation_size = env.get_obs_size()
        self.observation_spaces = {
            name: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=-1,
                        high=1,
                        shape=(observation_size,),
                        dtype="float32",
                    ),
                    "action_mask": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.action_spaces[name].n,),
                        dtype=np.int8,
                    ),
                }
            )
            for name in self.agents
        }
        state_size = env.get_state_size()
        self.state_space = spaces.Box(low=-1, high=1, shape=(state_size,), dtype="float32")
        self._reward = 0

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _init_agents(self):
        last_type = ""
        agents = []
        action_spaces = {}
        self.agents_id = {}
        i = 0
        for agent_id, agent_info in self.env.agents.items():
            unit_action_space = spaces.Discrete(
                self.env.get_total_actions() - 1
            )  # no-op in dead units is not an action
            if agent_info.unit_type == self.env.marine_id:
                agent_type = "marine"
            elif agent_info.unit_type == self.env.marauder_id:
                agent_type = "marauder"
            elif agent_info.unit_type == self.env.medivac_id:
                agent_type = "medivac"
            elif agent_info.unit_type == self.env.hydralisk_id:
                agent_type = "hydralisk"
            elif agent_info.unit_type == self.env.zergling_id:
                agent_type = "zergling"
            elif agent_info.unit_type == self.env.baneling_id:
                agent_type = "baneling"
            elif agent_info.unit_type == self.env.stalker_id:
                agent_type = "stalker"
            elif agent_info.unit_type == self.env.colossus_id:
                agent_type = "colossus"
            elif agent_info.unit_type == self.env.zealot_id:
                agent_type = "zealot"
            else:
                raise AssertionError(f"agent type {agent_type} not supported")

            if agent_type == last_type:
                i += 1
            else:
                i = 0

            agents.append(f"{agent_type}_{i}")
            self.agents_id[agents[-1]] = agent_id
            action_spaces[agents[-1]] = unit_action_space
            last_type = agent_type

        return agents, action_spaces

    def seed(self, seed=None):
        if seed is None:
            self.env._seed = seeding.create_seed(seed, max_bytes=4)
        else:
            self.env._seed = seed
        self.env.full_restart()

    def render(self, mode="human"):
        self.env.render(mode)

    def close(self):
        self.env.close()

    def reset(self, seed=None, options=None):
        self.env._episode_count = 1
        self.env.reset()

        self.agents = self.possible_agents[:]
        self.frames = 0
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        return self._observe_all()

    def get_agent_smac_id(self, agent):
        return self.agents_id[agent]

    def _all_rewards(self, reward):
        all_rewards = [reward] * len(self.agents)
        return {
            agent: reward for agent, reward in zip(self.agents, all_rewards)
        }

    def _observe_all(self):
        all_obs = []
        for agent in self.agents:
            agent_id = self.get_agent_smac_id(agent)
            obs = self.env.get_obs_agent(agent_id)
            action_mask = self.env.get_avail_agent_actions(agent_id)
            action_mask = action_mask[1:]
            action_mask = np.array(action_mask).astype(np.int8)
            obs = np.asarray(obs, dtype=np.float32)
            all_obs.append({"observation": obs, "action_mask": action_mask})
        return {agent: obs for agent, obs in zip(self.agents, all_obs)}

    def _all_terms_truncs(self, terminated=False, truncated=False):
        terminations = [True] * len(self.agents)

        if not terminated:
            for i, agent in enumerate(self.agents):
                agent_done = False
                agent_id = self.get_agent_smac_id(agent)
                agent_info = self.env.get_unit_by_id(agent_id)
                if agent_info.health == 0:
                    agent_done = True
                terminations[i] = agent_done

        terminations = {a: bool(t) for a, t in zip(self.agents, terminations)}
        truncations = {a: truncated for a in self.agents}

        return terminations, truncations

    def step(self, all_actions):
        action_list = [0] * self.env.n_agents
        for agent in self.agents:
            agent_id = self.get_agent_smac_id(agent)
            if agent in all_actions:
                if all_actions[agent] is None:
                    action_list[agent_id] = 0
                else:
                    action_list[agent_id] = all_actions[agent] + 1
        self._reward, terminated, smac_info = self.env.step(action_list)
        self.frames += 1

        all_infos = {agent: {} for agent in self.agents}
        # all_infos.update(smac_info)
        all_terms, all_truncs = self._all_terms_truncs(
            terminated=terminated, truncated=(self.frames >= self.max_cycles)
        )
        all_rewards = self._all_rewards(self._reward)
        all_observes = self._observe_all()

        self.agents = [agent for agent in self.agents if not all_terms[agent]]
        self.agents = [agent for agent in self.agents if not all_truncs[agent]]

        return all_observes, all_rewards, all_terms, all_truncs, all_infos
    
    def state(self):
        return self.env.get_state()

    def __del__(self):
        self.env.close()


env = make_env(raw_env)


class _parallel_env(smac_parallel_env, EzPickle):
    metadata = {"render.modes": ["human"], "name": "sc2"}

    def __init__(self, max_cycles, **smac_args):
        EzPickle.__init__(self, max_cycles, **smac_args)
        env = StarCraft2Env(**smac_args)
        super().__init__(env, max_cycles)
