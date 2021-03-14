import time
import random
import numpy as np

from smac_env import sc2_v0

def random_demo(env, render=True, cycles=100):
    '''
    Runs an env object with random actions.
    '''

    total_reward = 0
    done = False

    for cycle in range(cycles):
        env.reset()
        for agent in env.agent_iter():

            if render:
                env.render()

            obs, reward, done, info = env.last()
            total_reward += reward
            if done:
                action = None
            elif isinstance(obs, dict) and 'action_mask' in obs:
                action = random.choice(np.flatnonzero(obs['action_mask']))
            else:
                action = env.action_spaces[agent].sample()
            env.step(action)

    print("Total reward", total_reward, "done", done)

    if render:
        env.render()
        time.sleep(2)

    env.close()

    return total_reward

def main():
    env = sc2_v0.env(map_name = "25m", step_mul=2, window_size_x=1000, window_size_y=1000)
    random_demo(env)

if __name__ == "__main__":
    main()
