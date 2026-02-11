# Tool Use Project - Windows Setup Guide

This guide explains how to build and run the Simulation Server, Visualizer, and RL Trainer on Windows.

## Prerequisites
* **Docker Desktop**: For running the Simulation Server.
* **Anaconda/Miniconda**: For the Python RL environment.
* **Visual Studio 2022**: Install the "Desktop development with C++" workload (needed for the Visualizer).
* **CMake**: Version 3.20 or higher.

## Component 1: Simulation Server (Docker)
The server runs in a container to ensure physics consistency.
1. Open a terminal in the project root.
2. Build the image:
    ```powershell
    docker build -f simulation/Dockerfile.server -t tool-sim-server ./simulation
    ```
3. Run the container:
    ```powershell
    docker run --rm -p 3000:3000 tool-sim-server
    ```

## Component 2: Python Environment (Conda)
1. Open Anaconda Prompt.
2. Create and activate the environment:
    ```powershell
    conda create -n robotics python=3.11 -y
    conda activate robotics
    ```
3. Install PyTorch (ensure this matches your GPU/CUDA version):
    ```powershell
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```
4. Install RL and Graph dependencies:
    ```powershell
    pip install tensordict==0.8.3 torchrl==0.8.1 --no-deps
    pip install torch-geometric
    pip install websocket-client websockets
    pip install importlib_metadata orjson
    pip install cloudpickle tqdm tensorboard
    ```

## Component 3: Visualizer (Native Windows)
The Visualizer uses Raylib and needs to run natively to access your GPU for the UI.
1. Open a terminal in simulation/.
2. Add missing libraries to conda environment
    ```powershell
    conda activate robotics
    conda install -c conda-forge zlib libuv
    ```
3. Generate build files:
    ```powershell
    mkdir build
    cd build
    Remove-Item * -Recurse -Force
    cmake .. -DCMAKE_PREFIX_PATH="$env:CONDA_PREFIX/Library" -DCMAKE_CXX_STANDARD=20
    ```
4. Run these commands in your terminal before building:
    ```powersheel
    $env:CL = "/I""$env:CONDA_PREFIX\Library\include"""
    $env:LINK = "/LIBPATH:""$env:CONDA_PREFIX\Library\lib"" uv.lib"
    ```
5. Build the project:
    ```powershell
    cmake --build . --config Release
    ```
The executables will be located in build/Release/.


## Execution Order
To see the agent training visually:
1. Start Server: ```docker run -p 3000:3000 tool-sim-server```
2. Activate Environment: ```conda activate robotics```
3. Start Relay: ```python relay.py``` (This writes the server data to ```last_scene.json```)
4. Start Visualizer: Run ```SimulationClient.exe```
5. Start Training: ```python -m model.train_ppo```

