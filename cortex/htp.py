import numpy as np
import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription
from cortex.hologram import HolographicSuperposition
import logging

logger = logging.getLogger("HTP_Protocol")

class HolographicTransferProtocol:
    """
    Zero-serialization cognitive syncing via phase-space resonance.
    Build it. Verify it. Harden it. Ship it.
    """
    def __init__(self, cortex_engine, hsm: HolographicSuperposition):
        self.cortex = cortex_engine
        self.hsm = hsm
        self.pc = RTCPeerConnection()
        self.channel = None

    async def setup_channel(self, is_offerer: bool = False):
        """Initializes the WebRTC Data Channel for UDP transmission."""
        if is_offerer:
            self.channel = self.pc.createDataChannel("htp_wave", ordered=False, maxRetransmits=0)
            self._bind_channel_events(self.channel)
        else:
            @self.pc.on("datachannel")
            def on_datachannel(channel):
                self.channel = channel
                self._bind_channel_events(self.channel)

    def _bind_channel_events(self, channel):
        @channel.on("message")
        async def on_message(message):
            # Expecting raw bytes: D floats for Hologram, D floats for Anchor
            data = np.frombuffer(message, dtype=np.float32)
            if len(data) != self.hsm.dims * 2:
                logger.error(f"[HTP] Invalid payload size received: {len(data)} != {self.hsm.dims * 2}")
                return
            
            hologram = data[:self.hsm.dims]
            context_anchor = data[self.hsm.dims:]
            await self.process_incoming_wave(hologram, context_anchor)

    async def transmit_wave(self, hot_nodes: list):
        """Superposes local semantic concepts and pulses the UDP wave."""
        if not self.channel or self.channel.readyState != "open":
            logger.warning("[HTP] Channel not ready.")
            return

        if not hot_nodes:
            return

        # 1. Generate the ephemeral transmission anchor
        context_anchor = np.random.uniform(0, 2*np.pi, self.hsm.dims).astype(np.float32)

        # 2. Bind semantics to the anchor (database agnostic)
        traces = [self.hsm.bind_to_anchor(node.hvec, context_anchor) for node in hot_nodes if node.hvec is not None]
        
        if not traces:
            return
            
        hologram = self.hsm.superpose_to_phase(traces).astype(np.float32)

        # 3. Concat to a single 2KB payload and transmit
        payload = np.concatenate((hologram, context_anchor)).tobytes()
        self.channel.send(payload)
        logger.info(f"[HTP] Transmitted wave: {len(payload)} bytes. Representing {len(traces)} nodes.")

    async def process_incoming_wave(self, hologram: np.ndarray, context_anchor: np.ndarray):
        """Resonance sequence: Unbinds the wave and heats matching local nodes."""
        logger.info("[HTP] Wave received. Computing resonance...")
        
        # 1. Unbind the wave to extract the noisy semantic superposition
        v_local = self.hsm.unbind_from_phase(hologram, context_anchor)
        
        # 2. Sweep across the active local substrate
        resonating_node_ids = await self.cortex.find_resonating_nodes(v_local, threshold=0.75)
        
        if resonating_node_ids:
            logger.info(f"[HTP] Resonance achieved. Heating {len(resonating_node_ids)} native nodes.")
            
            # 3. Inject thermal heat directly into matching nodes using psycopg parameter array format
            async with self.cortex._pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        "UPDATE memories SET importance = LEAST(1.0, importance + 0.5) WHERE id = ANY($1::uuid[])",
                        resonating_node_ids
                    )
        else:
            logger.info("[HTP] No resonance. Structural confusion state triggered.")
