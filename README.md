# RL Tool Designer

Big thanks to **Marcel Lorenz** and **Shakhriyor Nizomov** for their contributions to the Python RL environment and the C++ simulation server, respectively.

This project implements a Reinforcement Learning (RL) pipeline where an agent learns to solve physics puzzles by **designing** and **placing** a tool. The goal is to create a static tool such that two balls, starting at different positions, eventually collide within a Box2D environment.

The system uses a **Server-Client architecture**: a high-speed C++ backend handles physics simulations via Box2D, while a Python RL pipeline manages the RL training loop using PyTorch, TorchRL, and PyTorch Geometric.

## 🚀 System Architecture

The project is divided into five main components:

1.  **Box2D Simulation Server**: Receives scene encodings via JSON, simulates the physics, and returns results.
2.  **TorchRL Environment**: Manages the two-phase task and communicates with the server via WebSockets.
3.  **Neural Networks**: An actor-critic architecture featuring a Scene Encoder, Designer and Placement heads.
4.  **RL Orchestrator**: Uses Proximal Policy Optimization to train the agent.
5.  **UI Visualizer**: Allows for real-time monitoring and manual stepping of the simulation.

## 📂 Project Structure

```text
├── model/                  # Python RL Logic
│   ├── env.py              # TorchRL Environment wrapper
│   ├── model_impl.py       # Neural Network architectures
│   └── train_ppo.py        # PPO Training script
├── simulation/             # C++ Physics Engine
│   ├── include/scene.h     # Box2D world definitions
│   ├── src/server.cpp      # Simulation server (WebSocket)
│   ├── src/client.cpp      # Raylib Visualizer
│   └── external/           # Box2D, Raylib, uWebSockets, ImGui
├── model_snapshots/        # Saved .pt model weights
├── scenes/                 # Scenario definitions and samplers
└── relay.py                # Bridge between Server and Visualizer
```

## 🛠️ Installation

### Prerequisites
*   **C++ Development**: CMake, a C++17 compiler (MSVC/GCC).
*   **Python**: Version 3.11+ recommended.
*   **Dependencies**: Box2D, Raylib, uWebSockets (included in `external/`).

### Setup
1.  **Python Environment**:
    ```bash
    conda create -n tool_use python=3.11
    conda activate tool_use
    pip install torch torchrl tensordict websockets
    ```
2.  **Build Simulation Server & Client**:
    ```bash
    cd simulation
    mkdir build && cd build
    cmake ..
    cmake --build . --config Release
    ```

## 🎮 How to Run

To visualize the training or test a trained model, follow this order:

### 1. Start the Simulation Server
This launches the physics engine on port 3000.
```bash
# In terminal 1
docker run -p 3000:3000 tool-sim-server
```

### 2. Start the Relay & Visualizer
The relay captures server broadcasts, and the client renders them.
```bash
# In terminal 2
python relay.py

# In terminal 3
.\simulation\build\Release\SimulationClient.exe
```

### 3. Start Training or Evaluation
Run the PPO orchestrator.
```bash
# To Train
python -m model.train_ppo

# To Visualize a Snapshot
python visualize_model.py --ckpt "model_snapshots/RUN_.../model_X.pt" --level X --count 10 --delay 10
```

---
*Created for the "ML in Robotics" course at HU Berlin.*