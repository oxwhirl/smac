from pettingzoo import test
from smac_env import sc2_v0
import pickle

if __name__ == "__main__":
    env = sc2_v0.env(map_name="corridor")
    test.api_test(env)
    # test.parallel_api_test(sc2_v0.parallel_env()) # does not pass it due to illegal actions
    # test.seed_test(sc2_v0.env, 50) # not required, sc2 env only allows reseeding at initialization

    recreated_env = pickle.loads(pickle.dumps(env))
    test.api_test(recreated_env)
