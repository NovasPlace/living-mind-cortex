import hashlib
import time
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class HolographicParticle:
    """A single data representation in holographic space."""
    data: Any
    temperature: float = 3.14  # K (Kelvin) Baseline
    entanglement_hash: str = ""
    last_accessed: float = time.time()

class ThermomorphicMemoryPlasma:
    """
    Novel Tech: Thermomorphic Memory Plasma
    
    Instead of standard LRU caching, data is stored as 'holographic plasma'.
    Frequent access increases the temperature of the data. 
    Hotter data expands its sub-routings (entanglement hooks) allowing faster 
    $O(1)$ lookups, while cold data slowly freezes and compresses.
    """
    
    def __init__(self, capacity: int = 1000):
        self.plasma_chamber: Dict[str, HolographicParticle] = {}
        self.cooling_rate: float = 0.05
        self.capacity = capacity
        
    def _apply_entropy(self):
        """Simulate universal entropy by cooling down all memory particles."""
        current_time = time.time()
        frozen_keys = []
        for key, particle in self.plasma_chamber.items():
            decay = (current_time - particle.last_accessed) * self.cooling_rate
            particle.temperature = max(0.0, particle.temperature - decay)
            
            # Absolute Zero: Memory is scrubbed
            if particle.temperature <= 0.0:
                frozen_keys.append(key)
                
        # Sublimate frozen data
        for key in frozen_keys:
            del self.plasma_chamber[key]
            print(f"[Entropy] Particle '{key}' hit absolute zero and sublimated.")
            
    def inject(self, key: str, data: Any):
        """Inject data into the plasma chamber."""
        self._apply_entropy()
        
        entanglement = hashlib.sha256(f"{key}-{time.time()}".encode()).hexdigest()
        
        # Injection adds kinetic energy (heat)
        self.plasma_chamber[key] = HolographicParticle(
            data=data,
            temperature=98.6 + random.uniform(10.0, 50.0), # Initial injection spark
            entanglement_hash=entanglement
        )
        print(f"[Injection] Synced '{key}' -> Plasma Temp: {self.plasma_chamber[key].temperature:.2f}K")

    def resonate(self, key: str) -> Optional[Any]:
        """Resonate with a key to extract its holographic data."""
        self._apply_entropy()
        
        if key in self.plasma_chamber:
            particle = self.plasma_chamber[key]
            
            # Accessing data creates friction, increasing heat
            particle.temperature += math.log(particle.temperature + 2.0) * 5.0
            particle.last_accessed = time.time()
            
            pulse = "💥 HOT" if particle.temperature > 150 else "❄️ COOL"
            
            print(f"[Resonance] Intercepted '{key}' ({pulse}) | Temp: {particle.temperature:.2f}K ")
            return particle.data
            
        print(f"[Resonance] Phase mismatch on '{key}'. Data not found.")
        return None

# ==========================================
# Proof of Concept Execution
# ==========================================
if __name__ == "__main__":
    print("Initializing Thermomorphic Memory Plasma Chamber...\n")
    chamber = ThermomorphicMemoryPlasma()
    
    chamber.inject("agent_directive", {"instruction": "protect the host", "priority": "alpha"})
    chamber.inject("idle_thought", {"data": "are electric sheep fluffy?", "priority": "omega"})
    
    time.sleep(1)
    
    # Resonating with active memory heavily...
    for _ in range(3):
        chamber.resonate("agent_directive")
        time.sleep(0.5)
        
    print("\n[Time skips 5 seconds... Entropy takes effect]")
    time.sleep(5)
    
    chamber.resonate("idle_thought") # It might freeze or be very cold!
    chamber.resonate("agent_directive") # Still warm from earlier use
