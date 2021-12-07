import numpy as np


def uniform_attack_distribution(
    n_agents, attack_probability_low=0.7, attack_probability_high=1.0
):
    rng = np.random.default_rng()

    def attack_distribution():
        return rng.uniform(
            low=attack_probability_low,
            high=attack_probability_high,
            size=(n_agents,),
        )

    return attack_distribution
