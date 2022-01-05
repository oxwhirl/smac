from typing import List, Optional
from itertools import combinations_with_replacement
from random import choice, shuffle

DISTRIBUTIONS = {}


def get_distribution_function(teammate_distribution: Optional[str]):
    return DISTRIBUTIONS[teammate_distribution]


def register_distribution(distribution_str: str, distribution_fn):
    DISTRIBUTIONS[distribution_str] = distribution_fn


def stub_distribution_function(
    ally_unit_types: List[str],
    n_ally_units: int,
    enemy_units: List[object],
    **kwargs
):
    """Stub distribution function used when the meta_marl setting of SMAC is
    turned off. Raises a NotImplementedError.

    Args:
        ally_unit_types (List[str]): A list of unit types available to the ally team
        n_ally_units (int): The number of ally units
        enemy_units (List[object]): The list of enemy units that will be faced. This can
            be used to generate a team that matches the enemies' capabilities.
    """

    def get_team():
        raise NotImplementedError("No distribution of teammates specified!")

    return get_team


register_distribution("stub", stub_distribution_function)


def all_teams_distribution_function(
    ally_unit_types: List[str],
    n_ally_units: int,
    enemy_units: List[object],
    **kwargs
):
    """Distribution function that just cycles through all possible ally team
    distributions. Args as specified in `stub_distribution_function`
    """
    all_combinations = list(
        combinations_with_replacement(ally_unit_types, n_ally_units)
    )

    def get_team():
        team = list(choice(all_combinations))
        shuffle(team)
        return team

    return get_team


register_distribution("all", all_teams_distribution_function)
