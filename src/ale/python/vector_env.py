from __future__ import annotations

from typing import Any

from gymnasium.core import RenderFrame, ActType, ObsType
from gymnasium.vector import VectorEnv
from gymnasium.vector.vector_env import ArrayType


class AtariVectorEnv(VectorEnv):

    def __init__(self,):
        pass

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        return super().reset(seed=seed, options=options)

    def step(self, actions: ActType) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        return super().step(actions)

    def recv(self):
        pass

    def render(self) -> tuple[RenderFrame, ...] | None:
        pass

    def close(self, **kwargs: Any):
        super().close(**kwargs)



