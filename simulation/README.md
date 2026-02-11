# Simulation environment for "Machine Learning in Robotics" Seminar

## Content
- WebSockets server
  - accepts incoming connections
  - waits for scene and tool definition
  - runs simulation, returns reward value
- Client
  - simple UI to debug/visualize scenes, tools and simulations

## Requirements/Installation
- a C++17 compiler (clang v21.1.6)
- CMake (v4.2.3)
- make

## Building
First, create a build directory:
```
$ mkdir build
```

Then, go into the build directory and create the build files:
```
$ cd build
$ cmake .. -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
```

Then, build:
```
$ make
```

You can then run the server with:
```
$ ./SimulationServer
```
This will setup a WebSockets server listening on port `3000`. Once the server is running, you can start connecting and sending scene definitions to it.

Similarly, you can run the client:
```
$ ./SimulationClient
```
Currently the client only helps visualize/debug scene definitions.

## Example

```py
from websockets.sync.client import connect
import json

def test():
    with connect("ws://localhost:3000") as websocket:
        scene_object1 = { "type": 0, "x":  0.0, "y":   0.0, "width": 16.0, "height":  0.5, "rotation":  0.0 }
        scene_object2 = { "type": 1, "x":  2.0, "y":   7.0, "width":  0.5, "height":  0.0, "rotation":  0.0 }
        scene_object3 = { "type": 1, "x": 14.0, "y":   7.0, "width":  0.5, "height":  0.0, "rotation":  0.0 }

        tool_segment1 = { "angle":  90, "length":  1.0 }
        tool_segment2 = { "angle":   0, "length":  5.0 }
        tool = { "x": 1.0, "y": 4.0, "segments": [tool_segment1, tool_segment2] }

        scene = {
            "scene": [scene_object1, scene_object2, scene_object3],
            "tool": tool
        }
        scene_json = json.dumps(scene)
        print(scene_json)

        websocket.send(scene_json)

        message = websocket.recv()
        websocket.close()

        print(f"Received: {message}")

test()
```

Once the simulation is complete, the server sends back the following result data structure:
```
{
  "completion": ..., # 0 if failed, 1 if completed
  "steps": ..., # the amount of simulation steps until it completed
  "maxSteps": ..., # max. amount of possible simulation steps
  "eucledianDistance": ..., # the eucledian distance between the two balls
  "actualDistance": ... # "actual" distance between the two balls (not yet implemented)
}
```

## External dependencies
- box2d
- ImGui
- nlohmann::json
- raylib
- rlImGui
- uSockets
- uWebSockets
