from dataclasses import dataclass
import random
from typing import List, Tuple, Union


class Samplable():
    def sample(self, rng: random.Random) -> float: ...


@dataclass(frozen=True)
class Const(Samplable):
    value: float

    def sample(self, rng: random.Random) -> float:
        return self.value
    

@dataclass(frozen=True)
class Uniform(Samplable):
    min: float
    max: float

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.min, self.max)
    

@dataclass(frozen=True)
class Gauss(Samplable):
    mean: float
    std: float

    def sample(self, rng: random.Random) -> float:
        return rng.gauss(self.mean, self.std)


@dataclass(frozen=True)
class Choice(Samplable):
    values: List[float]

    def sample(self, rng: random.Random) -> float:
        return rng.choice(self.values)


@dataclass(frozen=True)
class WorldSpec:
    width: Samplable
    height: Samplable


@dataclass(frozen=True)
class BallSpec:
    name: str
    x: Samplable
    y: Samplable
    radius: Samplable


@dataclass(frozen=True)
class WallSpec:
    name: str
    x: Samplable
    y: Samplable
    width: Samplable
    height: Samplable
    rotation: Samplable


@dataclass(frozen=True)
class Goal:
    a: str
    b: str


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    world: WorldSpec
    objects: List[Union[WallSpec, BallSpec]]
    goal: List[Goal]
    max_tries: int


@dataclass
class WorldState:
    width: float
    height: float


@dataclass
class BallState:
    x: float
    y: float
    radius: float


@dataclass
class WallState:
    x: float
    y: float
    width: float
    height: float
    rotation: float


@dataclass
class SampledScene:
    scenario: str
    world: WorldState
    scene: List[Union[WallState, BallState]]
    goal_pairs: List[Tuple[int, int]] # goal tuples, indices in scene array