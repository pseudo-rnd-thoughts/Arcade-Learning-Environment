---
title: Othello
---

# Othello

```{figure} ../_static/videos/environments/othello.gif
:width: 120px
:name: Othello
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|                   |                                   |
|-------------------|-----------------------------------|
| Make              | gymnasium.make("ALE/Othello-v5")  |
| Action Space      | Discrete(10)                      |
| Observation Space | Box(0, 255, (210, 160, 3), uint8) |

For more Othello variants with different observation and action spaces, see the variants section.

## Description

Take turns placing tiles of your color (white or black) on a grid. You can surround an opponents tiles to change their color to yours. The goals is to end the game with the most tiles of your color on the board.

For a more detailed documentation, see [the AtariAge page](https://atariage.com/manual_html_page.php?SoftwareLabelID=335)

## Actions

Othello has the action space of `Discrete(10)` with the table below listing the meaning of each action's meanings.
To enable all 18 possible actions that can be performed on an Atari 2600, specify `full_action_space=True` during
initialization or by passing `full_action_space=True` to `gymnasium.make`.

|   Value | Meaning   |
|---------|-----------|
|       0 | NOOP      |
|       1 | FIRE      |
|       2 | UP        |
|       3 | RIGHT     |
|       4 | LEFT      |
|       5 | DOWN      |
|       6 | UPRIGHT   |
|       7 | UPLEFT    |
|       8 | DOWNRIGHT |
|       9 | DOWNLEFT  |

See [environment specification](../env-spec) to see more information on the action meaning.

## Observations

Atari environments have three possible observation types:

- `obs_type="rgb"` -> `observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram"` -> `observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale"` -> `Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the q"rgb" type

See variants section for the type of observation used by each environment id by default.

## Variants

Othello has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

| Env-id         | obs_type=   | frameskip=   | repeat_action_probability=   |
|----------------|-------------|--------------|------------------------------|
| ALE/Othello-v5 | `rgb`       | `4`          | `0.25`                       |

See the [version history page](https://ale.farama.org/environments/#version-history-and-naming-schemes) to implement previously implemented environments, e.g., `OthelloNoFrameskip-v4`.

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

| Available Modes   | Default Mode   | Available Difficulties   | Default Difficulty   |
|-------------------|----------------|--------------------------|----------------------|
| `[0, 1, 2]`       | `0`            | `[0, 2]`                 | `0`                  |

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frame-skipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
