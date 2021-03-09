import time
import random
import numpy as np

from smac.examples.pettingzoo import sc2_v0


def random_demo(env, render=True, cycles=100):
    '''
    Runs an env object with random actions.
    '''

    env.reset()

    if hasattr(env, 'display_wait'):
        display_wait = env.display_wait
    else:
        display_wait = 0.0

    total_reward = 0
    done = False

    for cycle in range(cycles):
        env.reset()
        for agent in env.agent_iter():
            if render:
                env.render()
                time.sleep(display_wait)

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
    env = sc2_v0.env(map_name = "MMM", step_mul=2)
    random_demo(env)

if __name__ == "__main__":
    main()
