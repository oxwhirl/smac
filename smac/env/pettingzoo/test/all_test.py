from smac.env.starcraft2.maps import smac_maps
from pysc2 import maps as pysc2_maps
from smac.env.pettingzoo import StarCraft2PZEnv as sc2
import pytest
from pettingzoo import test
import pickle

smac_map_registry = smac_maps.get_smac_map_registry()
all_maps = pysc2_maps.get_maps()
map_names = []
for map_name in smac_map_registry.keys():
    map_class = all_maps[map_name]
    if map_class.path:
        map_names.append(map_name)


@pytest.mark.parametrize(("map_name"), map_names)
def test_env(map_name):
    env = sc2.env(map_name=map_name)
    test.api_test(env)
    # test.parallel_api_test(sc2_v0.parallel_env()) # does not pass it due to
    # illegal actions test.seed_test(sc2.env, 50) # not required, sc2 env only
    # allows reseeding at initialization
    test.render_test(env)

    recreated_env = pickle.loads(pickle.dumps(env))
    test.api_test(recreated_env)
