"""
Websocket Event Manager
Handles realtime bi-directional communication with the Tree of Life UI.
"""

from fastapi import WebSocket
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_pulse(self, data: dict):
        """Send pulse vitals to all connected UIs"""
        message = json.dumps({"type": "pulse", "data": data})
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass
                
    async def broadcast_event(self, event_type: str, message: str):
        """Send a specific organ firing event"""
        payload = json.dumps({"type": "event", "event": event_type, "message": message})
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                pass

manager = ConnectionManager()
