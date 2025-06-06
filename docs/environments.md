---
firstpage:
lastpage:
---

# Environments

```{toctree}
:hidden:
environments/adventure
environments/air_raid
environments/alien
environments/amidar
environments/assault
environments/asterix
environments/asteroids
environments/atlantis
environments/atlantis2
environments/backgammon
environments/bank_heist
environments/basic_math
environments/battle_zone
environments/beam_rider
environments/berzerk
environments/blackjack
environments/bowling
environments/boxing
environments/breakout
environments/carnival
environments/casino
environments/centipede
environments/chopper_command
environments/crazy_climber
environments/crossbow
environments/darkchambers
environments/defender
environments/demon_attack
environments/donkey_kong
environments/double_dunk
environments/earthworld
environments/elevator_action
environments/enduro
environments/entombed
environments/et
environments/fishing_derby
environments/flag_capture
environments/freeway
environments/frogger
environments/frostbite
environments/galaxian
environments/gopher
environments/gravitar
environments/hangman
environments/haunted_house
environments/hero
environments/human_cannonball
environments/ice_hockey
environments/jamesbond
environments/journey_escape
environments/kaboom
environments/kangaroo
environments/keystone_kapers
environments/king_kong
environments/klax
environments/koolaid
environments/krull
environments/kung_fu_master
environments/laser_gates
environments/lost_luggage
environments/mario_bros
environments/miniature_golf
environments/montezuma_revenge
environments/mr_do
environments/ms_pacman
environments/name_this_game
environments/othello
environments/pacman
environments/phoenix
environments/pitfall
environments/pitfall2
environments/pong
environments/pooyan
environments/private_eye
environments/qbert
environments/riverraid
environments/road_runner
environments/robotank
environments/seaquest
environments/sir_lancelot
environments/skiing
environments/solaris
environments/space_invaders
environments/space_war
environments/star_gunner
environments/superman
environments/surround
environments/tennis
environments/tetris
environments/tic_tac_toe_3d
environments/time_pilot
environments/trondead
environments/turmoil
environments/tutankham
environments/up_n_down
environments/venture
environments/video_checkers
environments/video_chess
environments/video_cube
environments/video_pinball
environments/wizard_of_wor
environments/word_zapper
environments/yars_revenge
environments/zaxxon
```

```{raw} html
   :file: environments/list.html
```

## Action Space

Each environment will use a sub-set of the full action space listed below:

| Value   | Meaning      | Value   | Meaning         | Value   | Meaning        |
|---------|--------------|---------|-----------------|---------|----------------|
| `0`     | `NOOP`       | `1`     | `FIRE`          | `2`     | `UP`           |
| `3`     | `RIGHT`      | `4`     | `LEFT`          | `5`     | `DOWN`         |
| `6`     | `UPRIGHT`    | `7`     | `UPLEFT`        | `8`     | `DOWNRIGHT`    |
| `9`     | `DOWNLEFT`   | `10`    | `UPFIRE`        | `11`    | `RIGHTFIRE`    |
| `12`    | `LEFTFIRE`   | `13`    | `DOWNFIRE`      | `14`    | `UPRIGHTFIRE`  |
| `15`    | `UPLEFTFIRE` | `16`    | `DOWNRIGHTFIRE` | `17`    | `DOWNLEFTFIRE` |

By default, most environments use a smaller subset of the legal actions excluding any actions that don't have an effect in the game.
If users are interested in using all possible actions, pass the keyword argument `full_action_space=True` to `gymnasium.make`.

## Continuous action space

The ALE supports continuous actions, parameterized as a 3-dimensional vector. The first two dimensions specify polar coordinates
(radius, theta), while the last dimension is the "fire" dimension. The ranges are:
1. **radius**: `[0.0, 1.0]`
2. **theta**: `[-np.pi, np.pi]`
3. **fire**: `[0.0, 1.0]`

Continuous action spaces still trigger the same events as the default discrete action space, but it is done via the parameter
`continuous_action_threshold` (i.e. if the "fire" dimension is above `continuous_action_threshold`, a "fire" event is triggered).
See [[3]](#3) for more details.

## Observation Space

The Atari environments observation can be
1. The RGB image that is displayed to a human player using `obs_type="rgb"` with observation space `Box(0, 255, (210, 160, 3), np.uint8)`
2. The grayscale version of the RGB image using `obs_type="grayscale"` with observation space `Box(0, 255, (210, 160), np.uint8)`
3. The RAM state (128 bytes) from the console using `obs_type="ram"` with observation space `Box(0, 255, (128), np.uint8)`

## Rewards

The exact reward dynamics depend on the environment and are usually documented in the game's manual. You can find these manuals on [AtariAge](https://atariage.com/).

## Stochasticity

As the Atari games are entirely deterministic, agents can achieve state-of-the-art performance by simply memorizing an optimal sequence of actions while completely ignoring observations from the environment.

To avoid this, there are several methods to avoid this.

1. Sticky actions: Instead of always simulating the action passed to the environment, there is a small
probability that the previously executed action is used instead. In the v0 and v5 environments, the probability of
repeating an action is `25%` while in v4 environments, the probability is `0%`. Users can specify the repeat action
probability using `repeat_action_probability` to `make`.
2. Frame-skipping: On each environment step, the action can be repeated for a random number of frames. This behavior
may be altered by setting the keyword argument `frameskip` to either a positive integer or
a tuple of two positive integers. If `frameskip` is an integer, frame skipping is deterministic, and in each step the action is
repeated `frameskip` many times. Otherwise, if `frameskip` is a tuple, the number of skipped frames is chosen uniformly at
random between `frameskip[0]` (inclusive) and `frameskip[1]` (exclusive) in each environment step.

## Common Arguments

When initializing Atari environments via `gymnasium.make`, you may pass some additional arguments. These work for any
Atari environment.

- **mode**: `int`. Game mode, see [[2]](#2). Legal values depend on the environment and are listed in the table above.

- **difficulty**: `int`. The difficulty of the game, see [[2]](#2). Legal values depend on the environment and are listed in
the table above. Together with `mode`, this determines the "flavor" of the game.

- **obs_type**: `str`. This argument determines what observations are returned by the environment. Its values are:
	- "ram": The 128 Bytes of RAM are returned
	- "rgb": An RGB rendering of the game is returned
	- "grayscale": A grayscale rendering is returned

- **frameskip**: `int` or a tuple of two `int`s. This argument controls stochastic frame skipping, as described in the section on stochasticity.

- **repeat_action_probability**: `float`. The probability that an action is repeated, also called "sticky actions", as described in the section on stochasticity.

- **full_action_space**: `bool`. If set to `True`, the action space consists of all legal actions on the console. Otherwise, the
action space will be reduced to a subset.

- **continuous**: `bool`. If set to True, will convert the action space into a Gymnasium [`Box`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Box) space.
Actions passed into the environment are then thresholded to discrete using the `continuous_action_threshold` parameter.

- **continuous_action_threshold**: `float`. This determines the threshold for actions to be thresholded into discrete actions.

- **render_mode**: `str`. Specifies the rendering mode. Its values are:
	- human: Display the screen and enable game sounds. This will lock emulation to the ROMs specified FPS
	- rgb_array: Returns the current environment RGB frame of the environment.

## Version History and Naming Schemes

In v0.11, the number of registered Atari environments was significantly reduced from 960 to 210 to only register `{rom_name}NoFrameskip-v4` the most popular environment and `ALE/{rom_name}-v5` following the best practices outlined in [[2]](#2).

| Name                    | `obs_type=` | `frameskip=` | `repeat_action_probability=` | `full_ation_space=` |
|-------------------------|-------------|--------------|------------------------------|---------------------|
| AdventureNoFrameskip-v4 | `"rgb"`     | `1`          | `0.00`                       | `False`             |
| ALE/Adventure-v5        | `"rgb"`     | `4`          | `0.25`                       | `False`             |

Importantly, `repeat_action_probability=0.25` can negatively impact the performance of agents so when comparing training graphs, be aware of the parameters used for fair comparisons.

To create previously implemented environment use the following parameters, `gymnasium.make(env_id, obs_type=..., frameskip=..., repeat_action_probability=..., full_action_space=...)`.

| Name                          | `obs_type=` | `frameskip=` | `repeat_action_probability=` | `full_action_space=` |
|-------------------------------|-------------|--------------|------------------------------|----------------------|
| Adventure-v0                  | `"rgb"`     | `(2, 5,)`    | `0.25`                       | `False`              |
| AdventureDeterministic-v0     | `"rgb"`     | `4`          | `0.25`                       | `False`              |
| AdventureNoframeskip-v0       | `"rgb"`     | `1`          | `0.25`                       | `False`              |
| Adventure-ram-v0              | `"ram"`     | `(2, 5,)`    | `0.25`                       | `False`              |
| Adventure-ramDeterministic-v0 | `"ram"`     | `4`          | `0.25`                       | `False`              |
| Adventure-ramNoframeskip-v0   | `"ram"`     | `1`          | `0.25`                       | `False`              |
| Adventure-v4                  | `"rgb"`     | `(2, 5,)`    | `0.0`                        | `False`              |
| AdventureDeterministic-v4     | `"rgb"`     | `4`          | `0.0`                        | `False`              |
| AdventureNoframeskip-v4       | `"rgb"`     | `1`          | `0.0`                        | `False`              |
| Adventure-ram-v4              | `"ram"`     | `(2, 5,)`    | `0.0`                        | `False`              |
| Adventure-ramDeterministic-v4 | `"ram"`     | `4`          | `0.0`                        | `False`              |
| Adventure-ramNoframeskip-v4   | `"ram"`     | `1`          | `0.0`                        | `False`              |
| ALE/Adventure-v5              | `"rgb"`     | `4`          | `0.25`                       | `False`              |
| ALE/Adventure-ram-v5          | `"ram"`     | `4`          | `0.25`                       | `False`              |

## Flavors

Some games allow the user to set a difficulty level and a game mode. Different modes/difficulties may have different
game dynamics and (if a reduced action space is used) different action spaces. We follow the convention of [[2]](#2) and
refer to the combination of difficulty level and game mode as a flavor of a game. The following table shows
the available modes and difficulty levels for different Atari games:

| Environment      | Possible Modes                                  |   Default Mode | Possible Difficulties   |   Default Difficulty |
|------------------|-------------------------------------------------|----------------|-------------------------|----------------------|
| Adventure        | [0, 1, 2]                                       |              0 | [0, 1, 2, 3]            |                    0 |
| AirRaid          | [1, ..., 8]                                     |              1 | [0]                     |                    0 |
| Alien            | [0, 1, 2, 3]                                    |              0 | [0, 1, 2, 3]            |                    0 |
| Amidar           | [0]                                             |              0 | [0, 3]                  |                    0 |
| Assault          | [0]                                             |              0 | [0]                     |                    0 |
| Asterix          | [0]                                             |              0 | [0]                     |                    0 |
| Asteroids        | [0, ..., 31, 128]                               |              0 | [0, 3]                  |                    0 |
| Atlantis         | [0, 1, 2, 3]                                    |              0 | [0]                     |                    0 |
| Atlantis2        | [0]                                             |              0 | [0]                     |                    0 |
| Backgammon       | [0]                                             |              0 | [3]                     |                    0 |
| BankHeist        | [0, 4, 8, 12, 16, 20, 24, 28]                   |              0 | [0, 1, 2, 3]            |                    0 |
| BasicMath        | [5, 6, 7, 8]                                    |              5 | [0, 2, 3]               |                    0 |
| BattleZone       | [1, 2, 3]                                       |              1 | [0]                     |                    0 |
| BeamRider        | [0]                                             |              0 | [0, 1]                  |                    0 |
| Berzerk          | [1, ..., 9, 16, 17, 18]                         |              1 | [0]                     |                    0 |
| Blackjack        | [0]                                             |              0 | [0, 1, 2, 3]            |                    0 |
| Bowling          | [0, 2, 4]                                       |              0 | [0, 1]                  |                    0 |
| Boxing           | [0]                                             |              0 | [0, 1, 2, 3]            |                    0 |
| Breakout         | [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]   |              0 | [0, 1]                  |                    0 |
| Carnival         | [0]                                             |              0 | [0]                     |                    0 |
| Casino           | [0, 2, 3]                                       |              0 | [0, 1, 2, 3]            |                    0 |
| Centipede        | [22, 86]                                        |             22 | [0]                     |                    0 |
| ChopperCommand   | [0, 2]                                          |              0 | [0, 1]                  |                    0 |
| CrazyClimber     | [0, 1, 2, 3]                                    |              0 | [0, 1]                  |                    0 |
| Crossbow         | [0, 2, 4, 6]                                    |              0 | [0, 1]                  |                    0 |
| Darkchambers     | [0]                                             |              0 | [0]                     |                    0 |
| Defender         | [1, ..., 9, 16]                                 |              1 | [0, 1]                  |                    0 |
| DemonAttack      | [1, 3, 5, 7]                                    |              1 | [0, 1]                  |                    0 |
| DonkeyKong       | [0]                                             |              0 | [0]                     |                    0 |
| DoubleDunk       | [0, ..., 15]                                    |              0 | [0]                     |                    0 |
| Earthworld       | [0]                                             |              0 | [0]                     |                    0 |
| ElevatorAction   | [0]                                             |              0 | [0]                     |                    0 |
| Enduro           | [0]                                             |              0 | [0]                     |                    0 |
| Entombed         | [0]                                             |              0 | [0, 2]                  |                    0 |
| Et               | [0, 1, 2]                                       |              0 | [0, 1, 2, 3]            |                    0 |
| FishingDerby     | [0]                                             |              0 | [0, 1, 2, 3]            |                    0 |
| FlagCapture      | [8, 9, 10]                                      |              8 | [0]                     |                    0 |
| Freeway          | [0, ..., 7]                                     |              0 | [0, 1]                  |                    0 |
| Frogger          | [0, 1, 2]                                       |              0 | [0, 1]                  |                    0 |
| Frostbite        | [0, 2]                                          |              0 | [0]                     |                    0 |
| Galaxian         | [1, ..., 9]                                     |              1 | [0, 1]                  |                    0 |
| Gopher           | [0, 2]                                          |              0 | [0, 1]                  |                    0 |
| Gravitar         | [0, 1, 2, 3, 4]                                 |              0 | [0]                     |                    0 |
| Hangman          | [0, 1, 2, 3]                                    |              0 | [0, 1]                  |                    0 |
| HauntedHouse     | [0, ..., 8]                                     |              0 | [0, 1]                  |                    0 |
| Hero             | [0, 1, 2, 3, 4]                                 |              0 | [0]                     |                    0 |
| HumanCannonball  | [0, ..., 7]                                     |              0 | [0, 1]                  |                    0 |
| IceHockey        | [0, 2]                                          |              0 | [0, 1, 2, 3]            |                    0 |
| Jamesbond        | [0, 1]                                          |              0 | [0]                     |                    0 |
| JourneyEscape    | [0]                                             |              0 | [0, 1]                  |                    0 |
| Kaboom           | [0]                                             |              0 | [0]                     |                    0 |
| Kangaroo         | [0, 1]                                          |              0 | [0]                     |                    0 |
| KeystoneKapers   | [0]                                             |              0 | [0]                     |                    0 |
| KingKong         | [0, 1, 2, 3]                                    |              0 | [0]                     |                    0 |
| Klax             | [0, 1, 2]                                       |              0 | [0]                     |                    0 |
| Koolaid          | [0]                                             |              0 | [0]                     |                    0 |
| Krull            | [0]                                             |              0 | [0]                     |                    0 |
| KungFuMaster     | [0]                                             |              0 | [0]                     |                    0 |
| LaserGates       | [0]                                             |              0 | [0]                     |                    0 |
| LostLuggage      | [0, 1]                                          |              0 | [0, 1]                  |                    0 |
| MarioBros        | [0, 2, 4, 6]                                    |              0 | [0]                     |                    0 |
| MiniatureGolf    | [0]                                             |              0 | [0, 1]                  |                    0 |
| MontezumaRevenge | [0]                                             |              0 | [0]                     |                    0 |
| MrDo             | [0, 1, 2, 3]                                    |              0 | [0]                     |                    0 |
| MsPacman         | [0, 1, 2, 3]                                    |              0 | [0]                     |                    0 |
| NameThisGame     | [8, 24, 40]                                     |              8 | [0, 1]                  |                    0 |
| Othello          | [0, 1, 2]                                       |              0 | [0, 2]                  |                    0 |
| Pacman           | [0, ..., 7]                                     |              0 | [0, 1]                  |                    0 |
| Phoenix          | [0]                                             |              0 | [0]                     |                    0 |
| Pitfall          | [0]                                             |              0 | [0]                     |                    0 |
| Pitfall2         | [0]                                             |              0 | [0]                     |                    0 |
| Pong             | [0, 1]                                          |              0 | [0, 1, 2, 3]            |                    0 |
| Pooyan           | [10, 30, 50, 70]                                |             10 | [0]                     |                    0 |
| PrivateEye       | [0, 1, 2, 3, 4]                                 |              0 | [0, 1, 2, 3]            |                    0 |
| Qbert            | [0]                                             |              0 | [0, 1]                  |                    0 |
| Riverraid        | [0]                                             |              0 | [0, 1]                  |                    0 |
| RoadRunner       | [0]                                             |              0 | [0]                     |                    0 |
| Robotank         | [0]                                             |              0 | [0]                     |                    0 |
| Seaquest         | [0]                                             |              0 | [0, 1]                  |                    0 |
| SirLancelot      | [0]                                             |              0 | [0]                     |                    0 |
| Skiing           | [0]                                             |              0 | [0]                     |                    0 |
| Solaris          | [0]                                             |              0 | [0]                     |                    0 |
| SpaceInvaders    | [0, ..., 15]                                    |              0 | [0, 1]                  |                    0 |
| SpaceWar         | [6, ..., 17]                                    |              6 | [0]                     |                    0 |
| StarGunner       | [0, 1, 2, 3]                                    |              0 | [0]                     |                    0 |
| Superman         | [0]                                             |              0 | [0, 1, 2, 3]            |                    0 |
| Surround         | [0, 2]                                          |              0 | [0, 1, 2, 3]            |                    0 |
| Tennis           | [0, 2]                                          |              0 | [0, 1, 2, 3]            |                    0 |
| Tetris           | [0]                                             |              0 | [0]                     |                    0 |
| TicTacToe3D      | [0, ..., 8]                                     |              0 | [0, 2]                  |                    0 |
| TimePilot        | [0]                                             |              0 | [0, 1, 2]               |                    0 |
| Trondead         | [0]                                             |              0 | [0, 1]                  |                    0 |
| Turmoil          | [0, ..., 8]                                     |              0 | [0]                     |                    0 |
| Tutankham        | [0, 4, 8, 12]                                   |              0 | [0]                     |                    0 |
| UpNDown          | [0]                                             |              0 | [0, 1, 2, 3]            |                    0 |
| Venture          | [0]                                             |              0 | [0, 1, 2, 3]            |                    0 |
| VideoCheckers    | [1, ..., 9, 11, ..., 19]                        |              1 | [0]                     |                    0 |
| VideoChess       | [0, 1, 2, 3, 4]                                 |              0 | [0]                     |                    0 |
| VideoCube        | [0, 1, 2, 100, 101, 102, ..., 5000, 5001, 5002] |              0 | [0, 1]                  |                    0 |
| VideoPinball     | [0, 2]                                          |              0 | [0, 1]                  |                    0 |
| WizardOfWor      | [0]                                             |              0 | [0, 1]                  |                    0 |
| WordZapper       | [0, ..., 23]                                    |              0 | [0, 1, 2, 3]            |                    0 |
| YarsRevenge      | [0, 32, 64, 96]                                 |              0 | [0, 1]                  |                    0 |
| Zaxxon           | [0, 8, 16, 24]                                  |              0 | [0]                     |                    0 |

## References

(#1)=
<a id="1">[1]</a>
MG Bellemare, Y Naddaf, J Veness, and M Bowling.
"The arcade learning environment: An evaluation platform for general agents."
Journal of Artificial Intelligence Research (2012).

(#2)=
<a id="2">[2]</a>
Machado et al.
"Revisiting the Arcade Learning Environment: Evaluation Protocols
and Open Problems for General Agents"
Journal of Artificial Intelligence Research (2018)
URL: https://jair.org/index.php/jair/article/view/11182

(#3)=
<a id="3">[3]</a>
Jesse Farebrother and Pablo Samuel Castro
"CALE: Continuous Arcade Learning Environment"
Advances in Neural Information Processing Systems (NeurIPS 2024)
URL: https://arxiv.org/abs/2410.23810
