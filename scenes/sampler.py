import math
from typing import Dict

from scenes.levels import TWO_BALLS_ON_FLOOR
from scenes.models import *


TYPE_WALL = 0
TYPE_BALL = 1


def circle_overlaps_rotated_rect(
    cx: float, cy: float, r: float,
    rx: float, ry: float, w: float, h: float,
    angle: float,   # radians CCW
) -> bool:
    # Transform circle center into rectangle-local coordinates by inverse-rotating
    # around the rect pivot (rx, ry).
    sx, sy = cx - rx, cy - ry
    ca, sa = math.cos(-math.radians(angle)), math.sin(-math.radians(angle))
    lx = sx * ca - sy * sa
    ly = sx * sa + sy * ca

    # Axis-aligned rect in local space: [0, w] x [0, h]
    closest_x = min(max(lx, 0.0), w)
    closest_y = min(max(ly, 0.0), h)

    dx = lx - closest_x
    dy = ly - closest_y

    # "touching" is NOT overlap => strict <
    return (dx * dx + dy * dy) < (r * r)


def has_invalid_overlaps(scene: List[Union[WallState, BallState]]) -> bool:
    balls = [o for o in scene if isinstance(o, BallState)]
    walls = [o for o in scene if isinstance(o, WallState)]

    # --- Ball vs Ball ---
    for i in range(len(balls)):
        b1 = balls[i]
        for j in range(i + 1, len(balls)):
            b2 = balls[j]

            dx = b1.x - b2.x
            dy = b1.y - b2.y
            r = b1.radius + b2.radius

            # "touching" is not overlap; use < not <=
            if dx*dx + dy*dy < r*r:
                return True

    # --- Ball vs Wall ---
    for b in balls:
        for w in walls:
            if circle_overlaps_rotated_rect(
                cx=b.x, cy=b.y, r=b.radius,
                rx=w.x, ry=w.y, w=w.width, h=w.height,
                angle=w.rotation,
            ):
                return True

    return False


class ScenarioSampler:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def sample(self, spec: ScenarioSpec) -> SampledScene:
        for _ in range(spec.max_tries):
            world = WorldState(
                width=spec.world.width.sample(self.rng),
                height=spec.world.height.sample(self.rng)
            )

            scene: List[Union[WallState, BallState]] = []
            name_to_idx: Dict[str, int] = {}

            for obj in spec.objects:                
                name_to_idx[obj.name] = len(scene)

                if isinstance(obj, WallSpec):
                    w = WallState(
                        x=obj.x.sample(self.rng),
                        y=obj.y.sample(self.rng),
                        width=obj.width.sample(self.rng),
                        height=obj.height.sample(self.rng),
                        rotation=obj.rotation.sample(self.rng)
                    )
                    scene.append(w)

                elif isinstance(obj, BallSpec):
                    b = BallState(
                        x=obj.x.sample(self.rng),
                        y=obj.y.sample(self.rng),
                        radius=obj.radius.sample(self.rng)
                    )
                    scene.append(b)

                else:
                    raise TypeError(obj)

            goals = [(name_to_idx[goal.a], name_to_idx[goal.b]) for goal in spec.goal]

            if (has_invalid_overlaps(scene)):
                continue

            return SampledScene(
                scenario=spec.name,
                world=world,
                scene=scene,
                goal_pairs=goals
            )
        
        raise RuntimeError(f"Failed to sample valid scene for {spec.name} after {spec.max_tries} tries")