import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from pettingzoo import test
import sc2
import pickle

if __name__ == "__main__":
    env = sc2.env(map_name="corridor")
    test.api_test(env)
    # test.parallel_api_test(sc2_v0.parallel_env()) # does not pass it due to illegal actions
    # test.seed_test(sc2_v0.env, 50) # not required, sc2 env only allows reseeding at initialization

    recreated_env = pickle.loads(pickle.dumps(env))
    test.api_test(recreated_env)
