import asyncio
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription

# --- MOCK CLASSES FOR STANDALONE TESTING ---
class MockHologram:
    def __init__(self, dim=256):
        self.dim = dim
    def bind_to_anchor(self, v, a): return (v + a) % (2*np.pi)
    def unbind(self, h, q): return (h - q) % (2*np.pi) # Simplified for mock
    def superpose_to_phase(self, traces): return traces[0] # Simplified for mock

class MockSubstrate:
    class MockEngine:
        async def find_resonating_nodes(self, wave, threshold):
            print(f"   -> [Engine] Sweeping wave. Shape: {wave.shape}")
            # Simulate a successful Subconscious Dredge
            return ["uuid-node-77", "uuid-node-88"]
    
    def __init__(self):
        self.engine = self.MockEngine()
        self.db = self

    async def transaction(self):
        class DummyTx:
            async def __aenter__(self): pass
            async def __aexit__(self, exc_type, exc, tb): pass
        return DummyTx()

    async def execute(self, query, *args):
        print(f"   -> [DB] Executing heat injection on nodes: {args[0]}")
# -------------------------------------------

async def signaling_loop(pc_a: RTCPeerConnection, pc_b: RTCPeerConnection, queue_a: asyncio.Queue, queue_b: asyncio.Queue):
    """Local SDP handshaking via memory queues."""
    # A creates offer
    offer = await pc_a.createOffer()
    await pc_a.setLocalDescription(offer)
    await queue_b.put(pc_a.localDescription)

    # B receives offer, creates answer
    desc = await queue_b.get()
    await pc_b.setRemoteDescription(desc)
    answer = await pc_b.createAnswer()
    await pc_b.setLocalDescription(answer)
    await queue_a.put(pc_b.localDescription)

    # A receives answer
    desc = await queue_a.get()
    await pc_a.setRemoteDescription(desc)

async def run_htp_simulation():
    print("=== IGNITING HTP INTEGRATION TEST ===")
    
    hsm = MockHologram(dim=256)
    
    pc_a = RTCPeerConnection()
    pc_b = RTCPeerConnection()
    queue_a = asyncio.Queue()
    queue_b = asyncio.Queue()

    # Setup channels
    channel_a = pc_a.createDataChannel("htp_wave", ordered=False, maxRetransmits=0)
    
    @pc_b.on("datachannel")
    def on_datachannel(channel_b):
        print("[Node B] Data channel established. Listening for waves...")
        @channel_b.on("message")
        def on_message(message):
            print(f"[Node B] Wave received! Payload size: {len(message)} bytes")
            data = np.frombuffer(message, dtype=np.float32)
            hologram = data[:hsm.dim]
            anchor = data[hsm.dim:]
            
            # Node B unbinds and sweeps
            v_local = hsm.unbind(hologram, anchor)
            
            # Fire the engine sweep (wrapping in task since we are in sync callback)
            substrate_b = MockSubstrate()
            asyncio.create_task(substrate_b.engine.find_resonating_nodes(v_local, 0.75))

    # Run signaling
    await signaling_loop(pc_a, pc_b, queue_a, queue_b)
    
    # Wait for ICE connection state
    await asyncio.sleep(1) 

    print("\n[Node A] Preparing semantic transmission...")
    # Generate mock hot nodes and anchor
    mock_semantics = [np.random.uniform(0, 2*np.pi, 256).astype(np.float32)]
    anchor = np.random.uniform(0, 2*np.pi, 256).astype(np.float32)
    
    traces = [hsm.bind_to_anchor(mock_semantics[0], anchor)]
    hologram = hsm.superpose_to_phase(traces).astype(np.float32)
    
    payload = np.concatenate((hologram, anchor)).tobytes()
    
    print(f"[Node A] Pulsing wave over UDP. {len(payload)} bytes.")
    channel_a.send(payload)

    # Let event loop process the message
    await asyncio.sleep(1)
    
    await pc_a.close()
    await pc_b.close()
    print("=== TEST COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(run_htp_simulation())
