from scenes.models import *


# TWO_BALLS_ON_FLOOR = ScenarioSpec(
#     name="TWO_BALLS_ON_FLOOR",
#     max_tries=1000,
#     world=WorldSpec(
#         width=Const(16.0),
#         height=Const(9.0)),
#     objects=[
#         WallSpec(
#             name="floor",
#             x=Const(0.0),
#             y=Const(0.0),
#             width=Const(16.0),
#             height=Uniform(0.5, 2.0),
#             rotation=Const(0.0)
#         ),
#         WallSpec(
#             name="wall_left",
#             x=Const(0.0),
#             y=Const(0.0),
#             width=Const(0.5),
#             height=Const(9.0),
#             rotation=Const(0.0)
#         ),
#         WallSpec(
#             name="wall_right",
#             x=Const(15.5),
#             y=Const(0.0),
#             width=Const(0.5),
#             height=Const(9.0),
#             rotation=Const(0.0)
#         ),
#         BallSpec(
#             name="ball1",
#             x=Uniform(1.0, 15.0),
#             y=Uniform(3.0, 5.0),
#             radius=Uniform(0.25, 0.75)
#         ),
#         BallSpec(
#             name="ball2",
#             x=Uniform(1.0, 15.0),
#             y=Uniform(3.0, 5.0),
#             radius=Uniform(0.25, 0.75)
#         )
#     ],
#     goal=[
#         Goal("ball1", "ball2")
#     ]
# )

TWO_BALLS_ON_FLOOR = ScenarioSpec(
    name="TWO_BALLS_ON_FLOOR",
    max_tries=1000,
    world=WorldSpec(width=Const(16.0), height=Const(9.0)),
    objects=[
        WallSpec(
            name="floor",
            x=Const(0.0), y=Const(0.0),
            width=Const(16.0), height=Uniform(0.5, 2.0),
            rotation=Const(0.0)
        ),
        WallSpec(
            name="wall_left",
            x=Const(0.0), y=Const(0.0),
            width=Const(0.5), height=Const(9.0),
            rotation=Const(0.0)
        ),
        WallSpec(
            name="wall_right",
            x=Const(15.5), y=Const(0.0),
            width=Const(0.5), height=Const(9.0),
            rotation=Const(0.0)
        ),
        BallSpec(
            name="ball1",
            x=Uniform(2.0, 14.0), y=Uniform(3.0, 5.0),
            radius=Uniform(0.25, 0.75)
        ),
        BallSpec(
            name="ball2",
            x=Uniform(2.0, 14.0), y=Uniform(3.0, 5.0),
            radius=Uniform(0.25, 0.75)
        )
    ],
    goal=[
        Goal("ball1", "ball2")
    ]
)

TWO_BALLS_ON_FLOOR_WITH_TINY_BUMP = ScenarioSpec(
    name="TWO_BALLS_ON_FLOOR_WITH_TINY_BUMP",
    max_tries=1000,
    world=WorldSpec(width=Const(16.0), height=Const(9.0)),
    objects=[
        WallSpec(
            name="floor",
            x=Const(0.0), y=Const(0.0),
            width=Const(16.0), height=Const(1.0),
            rotation=Const(0.0)
        ),
        WallSpec(
            name="wall_left",
            x=Const(0.0), y=Const(0.0),
            width=Const(0.5), height=Const(9.0),
            rotation=Const(0.0)
        ),
        WallSpec(
            name="wall_right",
            x=Const(15.5), y=Const(0.0),
            width=Const(0.5), height=Const(9.0),
            rotation=Const(0.0)
        ),
        WallSpec(
            name="bump",
            x=Const(7.0), y=Const(0.0),
            width=Const(2.0), height=Const(1.5),
            rotation=Const(0.0)
        ),
        BallSpec(
            name="ball1",
            x=Uniform(2.0, 14.0), y=Uniform(5.0, 7.0),
            radius=Uniform(0.25, 0.75)
        ),
        BallSpec(
            name="ball2",
            x=Uniform(2.0, 14.0), y=Uniform(5.0, 7.0),
            radius=Uniform(0.25, 0.75)
        )
    ],
    goal=[
        Goal("ball1", "ball2")
    ]
)

TWO_BALLS_ON_FLOOR_WITH_CENTER_WALL = ScenarioSpec(
    name="TWO_BALLS_ON_FLOOR_WITH_CENTER_WALL",
    max_tries=1000,
    world=WorldSpec(width=Const(16.0), height=Const(9.0)),
    objects=[
        WallSpec(
            name="floor",
            x=Const(0.0), y=Const(0.0),
            width=Const(16.0), height=Const(1.0),
            rotation=Const(0.0)
        ),
        WallSpec(
            name="wall_left",
            x=Const(0.0), y=Const(0.0),
            width=Const(0.5), height=Const(9.0),
            rotation=Const(0.0)
        ),
        WallSpec(
            name="wall_right",
            x=Const(15.5), y=Const(0.0),
            width=Const(0.5), height=Const(9.0),
            rotation=Const(0.0)
        ),
        WallSpec(
            name="wall_center",
            x=Const(7.0), y=Const(0.0),
            width=Const(2.0), height=Const(3),
            rotation=Const(0.0)
        ),
        BallSpec(
            name="ball1",
            x=Uniform(2.0, 6.0), y=Uniform(5.0, 7.0),
            radius=Uniform(0.25, 0.75)
        ),
        BallSpec(
            name="ball2",
            x=Uniform(10.0, 14.0), y=Uniform(5.0, 7.0),
            radius=Uniform(0.25, 0.75)
        )
    ],
    goal=[
        Goal("ball1", "ball2")
    ]
)


##### NOT PRESENT DURING TRAINING #####
# Requires building a bridge or pushing balls across a hole
TWO_BALLS_GAP = ScenarioSpec(
    name="TWO_BALLS_GAP",
    max_tries=1000,
    world=WorldSpec(width=Const(16.0), height=Const(9.0)),
    objects=[
        WallSpec(name="platform_left", x=Const(0.0), y=Const(0.0), width=Const(6.0), height=Const(2.0), rotation=Const(0.0)),
        WallSpec(name="platform_right", x=Const(10.0), y=Const(0.0), width=Const(6.0), height=Const(2.0), rotation=Const(0.0)),
        BallSpec(
            name="ball1", 
            x=Uniform(1.0, 4.0), y=Const(3.0), 
            radius=Const(0.5)
        ),
        BallSpec(
            name="ball2",
            x=Uniform(12.0, 15.0), y=Const(3.0), 
            radius=Const(0.5)
        )
    ],
    goal=[Goal("ball1", "ball2")]
)

# One ball is high up, the other is low.
TWO_BALLS_CLIFF = ScenarioSpec(
    name="TWO_BALLS_CLIFF",
    max_tries=1000,
    world=WorldSpec(width=Const(16.0), height=Const(9.0)),
    objects=[
        WallSpec(name="low_floor", x=Const(0.0), y=Const(0.0), width=Const(16.0), height=Const(1.0), rotation=Const(0.0)),
        WallSpec(
            name="cliff", 
            x=Const(0.0), y=Const(1.0), 
            width=Uniform(5.0, 8.0), height=Uniform(3.0, 5.0), 
            rotation=Const(0.0)
        ),
        BallSpec(
            name="ball1", 
            x=Uniform(1.0, 4.0), y=Const(7.0), 
            radius=Const(0.5)
        ),
        BallSpec(
            name="ball2", 
            x=Uniform(10.0, 14.0), y=Const(2.0), 
            radius=Const(0.5)
        )
    ],
    goal=[Goal("ball1", "ball2")]
)

# Two slanted floors meeting in the middle
TWO_BALLS_VALLEY = ScenarioSpec(
    name="TWO_BALLS_VALLEY",
    max_tries=1000,
    world=WorldSpec(width=Const(16.0), height=Const(9.0)),
    objects=[
        WallSpec(name="slope_left", x=Const(0.0), y=Const(2.0), width=Const(8.0), height=Const(0.5), rotation=Const(-15.0)),
        WallSpec(name="slope_right", x=Const(8.0), y=Const(0.0), width=Const(8.0), height=Const(0.5), rotation=Const(15.0)),
        BallSpec(
            name="ball1", 
            x=Uniform(1.0, 3.0), y=Const(5.0), 
            radius=Const(0.4)
        ),
        BallSpec(
            name="ball2", 
            x=Uniform(13.0, 15.0), y=Const(5.0), 
            radius=Const(0.4)
        )
    ],
    goal=[Goal("ball1", "ball2")]
)