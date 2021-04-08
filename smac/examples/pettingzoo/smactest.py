import time
import random
import numpy as np
from pettingzoo.utils import random_demo

from smac_env import sc2_v0

def main():
    env = sc2_v0.env(map_name="corridor", step_mul=2, window_size_x=800, window_size_y=800)
    random_demo(env)

if __name__ == "__main__":
    main()
