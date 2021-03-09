from smac.env import StarCraft2Env
from .smac_env import smac_parallel_env, make_env
from pettingzoo.utils.conversions import from_parallel_wrapper
from pettingzoo.utils.conversions import parallel_wrapper_fn
from .render import Renderer
from gym.utils import EzPickle

max_cycles_default = 1000

def parallel_env(max_cycles=max_cycles_default, **smac_args):
    return _parallel_env(max_cycles, **smac_args)

def raw_env(max_cycles=max_cycles_default, **smac_args):
    return from_parallel_wrapper(parallel_env(max_cycles, **smac_args))


env = make_env(raw_env)


class _parallel_env(smac_parallel_env, EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array'], 'name': "sc2_v0"}

    def __init__(self, max_cycles, **smac_args):
        EzPickle.__init__(self, max_cycles, **smac_args)
        env = StarCraft2Env(**smac_args)
        super().__init__(env, max_cycles)

