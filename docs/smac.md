## Table of Contents

- [StarCraft II](#starcraft-ii)
    - [Micromanagement](#micromanagement)
- [SMAC](#smac)
    - [Scenarios](#scenarios)
    - [State and Observations](#state-and-observations)
    - [Action Space](#action-space)
    - [Rewards](#rewards)
    - [Environment Settings](#environment-settings)

## StarCraft II

SMAC is based on the popular real-time strategy (RTS) game [StarCraft II](http://us.battle.net/sc2/en/game/guide/whats-sc2) written by [Blizzard](http://blizzard.com/).
In a regular full game of StarCraft II, one or more humans compete against each other or against a built-in game AI to gather resources, construct buildings, and build armies of units to defeat their opponents.

Akin to most RTSs, StarCraft has two main gameplay components: macromanagement and micromanagement. 
- _Macromanagement_ (macro) refers to high-level strategic considerations, such as economy and resource management. 
- _Micromanagement_ (micro) refers to fine-grained control of individual units.

### Micromanagement

StarCraft has been used as a research platform for AI, and more recently, RL. Typically, the game is framed as a competitive problem: an agent takes the role of a human player, making macromanagement decisions and performing micromanagement as a puppeteer that issues orders to individual units from a centralised controller.

In order to build a rich multi-agent testbed, we instead focus solely on micromanagement.
Micro is a vital aspect of StarCraft gameplay with a high skill ceiling, and is practiced in isolation by amateur and professional players.
For SMAC, we leverage the natural multi-agent structure of micromanagement by proposing a modified version of the problem designed specifically for decentralised control.
In particular, we require that each unit be controlled by an independent agent that conditions only on local observations restricted to a limited field of view centred on that unit.
Groups of these agents must be trained to solve challenging combat scenarios, battling an opposing army under the centralised control of the game's built-in scripted AI.

Proper micro of units during battles will maximise the damage dealt to enemy units while minimising damage received, and requires a range of skills.
For example, one important technique is _focus fire_, i.e., ordering units to jointly attack and kill enemy units one after another. When focusing fire, it is important to avoid _overkill_: inflicting more damage to units than is necessary to kill them.

Other common micromanagement techniques include: assembling units into formations based on their armour types, making enemy units give chase while maintaining enough distance so that little or no damage is incurred (_kiting_), coordinating the positioning of units to attack from different directions or taking advantage of the terrain to defeat the enemy. 

Learning these rich cooperative behaviours under partial observability is challenging task, which can be used to evaluate the effectiveness of multi-agent reinforcement learning (MARL) algorithms.

## SMAC

SMAC uses the [StarCraft II Learning Environment](https://github.com/deepmind/pysc2) to introduce a cooperative MARL environment.

### Scenarios

SMAC consists of a set of StarCraft II micro scenarios which aim to evaluate how well independent agents are able to learn coordination to solve complex tasks. 
These scenarios are carefully designed to necessitate the learning of one or more micromanagement techniques to defeat the enemy.
Each scenario is a confrontation between two armies of units.
The initial position, number, and type of units in each army varies from scenario to scenario, as does the presence or absence of elevated or impassable terrain.

The first army is controlled by the learned allied agents.
The second army consists of enemy units controlled by the built-in game AI, which uses carefully handcrafted non-learned heuristics.
At the beginning of each episode, the game AI instructs its units to attack the allied agents using its scripted strategies.
An episode ends when all units of either army have died or when a pre-specified time limit is reached (in which case the game is counted as a defeat for the allied agents).
The goal for each scenario is to maximise the win rate of the learned policies, i.e., the expected ratio of games won to games played.
To speed up learning, the enemy AI units are ordered to attack the agents' spawning point in the beginning of each episode.

Perhaps the simplest scenarios are _symmetric_ battle scenarios.
The most straightforward of these scenarios are _homogeneous_, i.e., each army is composed of only a single unit type (e.g., Marines).
A winning strategy in this setting would be to focus fire, ideally without overkill.
_Heterogeneous_ symmetric scenarios, in which there is more than a single unit type on each side (e.g., Stalkers and Zealots), are more difficult to solve.
Such challenges are particularly interesting when some of the units are extremely effective against others (this is known as _countering_), for example, by dealing bonus damage to a particular armour type.
In such a setting, allied agents must deduce this property of the game and design an intelligent strategy to protect teammates vulnerable to certain enemy attacks.

SMAC also includes more challenging scenarios, for example, in which the enemy army outnumbers the allied army by one or more units. In such _asymmetric_ scenarios it is essential to consider the health of enemy units in order to effectively target the desired opponent.

Lastly, SMAC offers a set of interesting _micro-trick_ challenges that require a higher-level of cooperation and a specific micromanagement trick to defeat the enemy. 
An example of a challenge scenario is _2m_vs_1z_ (aka Marine Double Team), where two Terran Marines need to defeat an enemy Zealot. In this setting, the Marines must design a strategy which does not allow the Zealot to reach them, otherwise they will die almost immediately. 
Another example is _so_many_banelings_ where 7 allied Zealots face 32 enemy Baneling units. Banelings attack by running against a target and exploding when reaching them, causing damage to a certain area around the target. Hence, if a large number of Banelings attack a handful of Zealots located close to each other, the Zealots will be defeated instantly. The optimal strategy, therefore, is to cooperatively spread out around the map far from each other so that the Banelings' damage is distributed as thinly as possible.
The _corridor_ scenario, in which 6 friendly Zealots face 24 enemy Zerglings, requires agents to make effective use of the terrain features. Specifically, agents should collectively wall off the choke point (the narrow region of the map) to block enemy attacks from different directions. Some of the micro-trick challenges are inspired by [StarCraft Master](http://us.battle.net/sc2/en/blog/4544189/new-blizzard-custom-game-starcraft-master-3-1-2012) challenge missions released by Blizzard.

The complete list of challenges is presented bellow. The difficulty of the game AI is set to _very difficult_ (7). Our experiments, however, suggest that this setting does significantly impact the unit micromanagement of the built-in heuristics.

| Name | Ally Units | Enemy Units | Type |
| :---: | :---: | :---: | :---:|
| 3m | 3 Marines | 3 Marines | homogeneous & symmetric |
| 8m | 8 Marines | 8 Marines | homogeneous & symmetric |
| 25m | 25 Marines | 25 Marines | homogeneous & symmetric |
| 2s3z |  2 Stalkers & 3 Zealots |  2 Stalkers & 3 Zealots | heterogeneous & symmetric |
| 3s5z |  3 Stalkers &  5 Zealots |  3 Stalkers &  5 Zealots | heterogeneous & symmetric |
| MMM |  1 Medivac, 2 Marauders & 7 Marines | 1 Medivac, 2 Marauders & 7 Marines | heterogeneous & symmetric |
| 5m_vs_6m | 5 Marines | 6 Marines | homogeneous & asymmetric |
| 8m_vs_9m  | 8 Marines | 9 Marines | homogeneous & asymmetric |
| 10m_vs_11m | 10 Marines | 11 Marines | homogeneous & asymmetric |
| 27m_vs_30m | 27 Marines | 30 Marines | homogeneous & asymmetric |
| 3s5z_vs_3s6z | 3 Stalkers & 5 Zealots | 3 Stalkers & 6 Zealots  | heterogeneous & asymmetric |
| MMM2 |  1 Medivac, 2 Marauders & 7 Marines |  1 Medivac, 3 Marauders & 8 Marines | heterogeneous & asymmetric |
| 2m_vs_1z | 2 Marines | 1 Zealot | micro-trick: alternating fire |
| 2s_vs_1sc| 2 Stalkers  | 1 Spine Crawler | micro-trick: alternating fire |
| 3s_vs_3z | 3 Stalkers | 3 Zealots | micro-trick: kiting |
|  3s_vs_4z | 3 Stalkers | 4 Zealots |  micro-trick: kiting |
| 3s_vs_5z | 3 Stalkers | 5 Zealots |  micro-trick: kiting |
| 6h_vs_8z | 6 Hydralisks  | 8 Zealots | micro-trick: focus fire |
| corridor | 6 Zealots  | 24 Zerglings | micro-trick: wall off |
| bane_vs_bane | 20 Zerglings & 4 Banelings  | 20 Zerglings & 4 Banelings | micro-trick: positioning |
| so_many_banelings| 7 Zealots  | 32 Banelings | micro-trick: positioning |
| 2c_vs_64zg| 2 Colossi  | 64 Zerglings | micro-trick: positioning |

### State and Observations

At each timestep, agents receive local observations drawn within their field of view. This  encompasses information about the map within a circular area around each unit and with a radius equal to the _sight range_. The sight range makes the environment partially observable from the standpoint of each agent. Agents can only observe other agents if they are both alive and located within the sight range. Hence, there is no way for agents to determine whether their teammates are far away or dead.

The feature vector observed by each agent contains the following attributes for both allied and enemy units within the sight range: _distance_, _relative x_, _relative y_, _health_, _shield_, and _unit\_type_ <sup>[1](#myfootnote1)</sup>. Shields serve as an additional source of protection that needs to be removed before any damage can be done to the health of units.
All Protos units have shields, which can regenerate if no new damage is dealt
(units of the other two races do not have this attribute).
In addition, agents have access to the last actions of allied units that are in the field of view. Lastly, agents can observe the terrain features surrounding them; particularly, the values of eight points at a fixed radius indicating height and walkability.

The global state, which is only available to agents during centralised training, contains information about all units on the map. Specifically, the state vector includes the coordinates of all agents relative to the centre of the map, together with unit features present in the observations. Additionally, the state stores the _energy_ of Medivacs and _cooldown_ of the rest of allied units, which represents the  minimum delay between attacks. Finally, the last actions of all agents are attached to the central state.

All features, both in the state as well as in the observations of individual agents, are normalised by their maximum values. The sight range is set to 9 for all agents.

### Action Space

The discrete set of actions which agents are allowed to take consists of _move[direction]_ (four directions: north, south, east, or west._, _attack[enemy_id]_, _stop_ and _no-op_. Dead agents can only take _no-op_ action while live agents cannot.
As healer units, Medivacs must use _heal[agent\_id]_ actions instead of _attack[enemy\_id]_. The maximum number of actions an agent can take ranges between 7 and 70, depending on the scenario.

To ensure decentralisation of the task, agents are restricted to use the _attack[enemy\_id]_ action only towards enemies in their _shooting range_.
This additionally constrains the unit's ability to use the built-in _attack-move_ macro-actions on the enemies far away. We set the shooting range equal to 6. Having a larger sight than shooting range forces agents to make use of move commands before starting to fire.

### Rewards

The overall goal is to have the highest win rate for each battle scenario.
We provide a corresponding option for _sparse rewards_, which will cause the environment to return only a reward of +1 for winning and -1 for losing an episode.
However, we also provide a default setting for a shaped reward signal calculated from the hit-point damage dealt and received by agents, some positive (negative) reward after having enemy (allied) units killed and/or a positive (negative) bonus for winning (losing) the battle.
The exact values and scales of this shaped reward can be configured using a range of flags, but we strongly discourage disingenuous engineering of the reward function (e.g. tuning different reward functions for different scenarios).

### Environment Settings

SMAC makes use of the [StarCraft II Learning Environment](https://arxiv.org/abs/1708.04782) (SC2LE) to communicate with the StarCraft II engine. SC2LE provides full control of the game by allowing to send commands and receive observations from the game. However, SMAC is conceptually different from the RL environment of SC2LE. The goal of SC2LE is to learn to play the full game of StarCraft II. This is a competitive task where a centralised RL agent receives RGB pixels as input and performs both macro and micro with the player-level control similar to human players. SMAC, on the other hand, represents a set of cooperative multi-agent micro challenges where each learning agent controls a single military unit.

SMAC uses the _raw API_ of SC2LE. Raw API observations do not have any graphical component and include information about the units on the map such as health, location coordinates, etc. The raw API also allows sending action commands to individual units using their unit IDs. This setting differs from how humans play the actual game, but is convenient for designing decentralised multi-agent learning tasks.

Since our micro-scenarios are shorter than actual StarCraft II games, restarting the game after each episode presents a computational bottleneck. To overcome this issue, we make use of the API's debug commands. Specifically, when all units of either army have been killed, we kill all remaining units by sending a debug action. Having no units left launches a trigger programmed with the StarCraft II Editor that re-spawns all units in their original location with full health, thereby restarting the scenario quickly and efficiently.

Furthermore, to encourage agents to explore interesting micro-strategies themselves, we limit the influence of the StarCraft AI on our agents. Specifically we disable the automatic unit attack against enemies that attack agents or that are located nearby. 
To do so, we make use of new units created with the StarCraft II Editor that are exact copies of existing units with two attributes modified: _Combat: Default Acquire Level_ is set to _Passive_ (default _Offensive_) and _Behaviour: Response_ is set to _No Response_ (default _Acquire_). These fields are only modified for allied units; enemy units are unchanged.

The sight and shooting range values might differ from the built-in _sight_ or _range_ attribute of some StarCraft II units. Our goal is not to master the original full StarCraft game, but rather to benchmark MARL methods for decentralised control.

<a name="myfootnote1">1</a>: _health_, _shield_ and _unit\_type_ of the unit the agent controls is also included in observations
