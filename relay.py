import asyncio
import websockets
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_PATH = os.path.join(BASE_DIR, "last_scene.json")

async def run_relay():
    uri = "ws://localhost:3000"
    print(f"Connecting to Server at {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send("IDENT_VIEWER")
            print("Connected and Identified as Viewer.")

            async for message in websocket:
                print(f"Message received ({len(message)} bytes)")
                
                try:
                    data = json.loads(message)
                    if "scene" in data:
                        with open(TARGET_PATH, "w") as f:
                            f.write(message)
                        print(f"  >>> UI Updated: {TARGET_PATH}")
                except Exception as e:
                    pass
    except Exception as e:
        print(f"Relay Error: {e}")

if __name__ == "__main__":
    print("Starting Relay Server...")
    target_dir = os.path.dirname(TARGET_PATH) or "."
    os.makedirs(target_dir, exist_ok=True)

    # Ensure directory exists
    try:
        with open(TARGET_PATH, "a"):
            pass
        print(f"Relay ensured file exists: {TARGET_PATH}")
    except OSError as e:
        print(f"ERROR: Cannot create target file {TARGET_PATH}: {e}")
        raise

    asyncio.run(run_relay())