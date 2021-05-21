import time
import random
import numpy as np
from pettingzoo.utils import random_demo

from smac.env.pettingzoo.smac_env import sc2

def main():
    env = sc2.env(map_name="corridor", step_mul=2, window_size_x=800, window_size_y=800)
    random_demo(env)

if __name__ == "__main__":
    main()
