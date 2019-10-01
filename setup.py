from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup

description = """SMAC - StarCraft Multi-Agent Challenge

SMAC offers a diverse set of decentralised micromanagement challenges based on
StarCraft II game. In these challenges, each of the units is controlled by an
independent, learning agent that has to act based only on local observations,
while the opponent's units are controlled by the built-in StarCraft II AI.

The accompanying paper which outlines the motivation for using SMAC as well as
results using the state-of-the-art deep multi-agent reinforcement learning
algorithms can be found at https://www.arxiv.link

Read the README at https://github.com/oxwhirl/smac for more information.
"""

setup(
    name='SMAC',
    version='0.1.0b1',
    description='SMAC - StarCraft Multi-Agent Challenge.',
    long_description=description,
    author='WhiRL',
    author_email='mikayel@samvelyan.com',
    license='MIT License',
    keywords='StarCraft, Multi-Agent Reinforcement Learning',
    url='https://github.com/oxwhirl/smac',
    packages=[
        'smac',
        'smac.env',
        'smac.env.starcraft2',
        'smac.env.starcraft2.maps',
        'smac.bin',
        'smac.examples',
        'smac.examples.rllib'
    ],
    install_requires=[
        'pysc2>=3.0.0',
        's2clientprotocol>=4.10.1.75800.0',
        'absl-py>=0.1.0',
        'numpy>=1.10',
    ],
)
