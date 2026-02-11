import asyncio
import websockets
import json

async def send_test():
    uri = "ws://localhost:3000"
    async with websockets.connect(uri) as ws:
        msg = {
            "scene": [
                {"type": 0, "x": 8, "y": 0.5, "width": 16, "height": 1, "rotation": 0}, # Floor
                {"type": 1, "x": 2, "y": 5, "width": 0.5, "height": 0.5, "rotation": 0}, # Ball A
                {"type": 1, "x": 14, "y": 5, "width": 0.5, "height": 0.5, "rotation": 0} # Ball B
            ],
            "tool": {
                "x": 8, "y": 4,
                "segments": [{"angle": 45, "length": 2.0}]
            }
        }
        await ws.send(json.dumps(msg))
        print("Sent valid test scene.")
        # Wait for result
        res = await ws.recv()
        print(f"Server result: {res}")

asyncio.run(send_test())