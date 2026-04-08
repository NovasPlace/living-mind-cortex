import time
import math
from typing import Any, Optional

class ThermomorphicMemoryPlasma:
    """
    Physics-based working memory layer using Newton's Law of Cooling.

    Each domain (e.g. 'code_expert', 'logic_expert') holds a temperature that
    decays exponentially over time. The router uses this temperature as a
    routing weight — hot domains stay loaded in VRAM, sublimated ones get evicted.

    Tuning knob:
        cooling_constant (k): Controls the "Focus Window" — how long a domain
        stays hot before sublimating. At k=0.005, a domain at 100K hits the
        sublimation threshold in ~921 seconds (15 min). Increase k for faster
        VRAM reclamation; decrease for stickier, longer sessions.

        Focus Window formula: t_sublimate = ln(100.0) / k
    """

    def __init__(self, sublimation_point: float = 0.0, cooling_constant: float = 0.005):
        # { domain_id: {'temp': float, 'last_seen': float, 'data': Any} }
        self.domains: dict = {}
        self.k = cooling_constant
        self.absolute_zero = sublimation_point

    def resonate(self, domain_id: str, friction_heat: float = 25.0, data: Any = None) -> float:
        """
        Heats up a domain by friction_heat. Persists current decayed state first
        to prevent double-decay on the next resonate() call.

        Args:
            domain_id:     The LoRA expert or memory zone identifier.
            friction_heat: Kinetic energy added per interaction (default 25K).
            data:          Optional payload to attach to this domain's particle.

        Returns:
            The new temperature after heating.
        """
        current_data = self._get_current_state(domain_id)
        new_temp = min(current_data['temp'] + friction_heat, 500.0)

        self.domains[domain_id] = {
            'temp': new_temp,
            'last_seen': time.time(),
            'data': data if data is not None else current_data.get('data'),
        }
        return new_temp

    def get_temp(self, domain_id: str) -> float:
        """
        Returns the current decayed temperature for a domain.
        Auto-evicts the domain if it crosses the sublimation threshold.
        """
        if domain_id not in self.domains:
            return self.absolute_zero

        state = self.domains[domain_id]
        elapsed = time.time() - state['last_seen']

        # Newton's Law of Cooling: T(t) = T_env + (T_0 - T_env) * e^(-k*t)
        decayed_temp = self.absolute_zero + (state['temp'] - self.absolute_zero) * math.exp(-self.k * elapsed)

        if decayed_temp < 1.0:  # Sublimation threshold — evict from registry
            del self.domains[domain_id]
            return self.absolute_zero

        return decayed_temp

    def get_data(self, domain_id: str) -> Optional[Any]:
        """Returns the payload stored in a domain's particle, or None if sublimated."""
        if domain_id not in self.domains or self.get_temp(domain_id) == self.absolute_zero:
            return None
        return self.domains[domain_id].get('data')

    def purge_frozen(self) -> list[str]:
        """
        Explicit sweep to evict all sublimated domains.
        Call this on a background tick for proactive VRAM cleanup.

        Returns:
            List of domain IDs that were evicted.
        """
        frozen = [d for d in list(self.domains) if self.get_temp(d) == self.absolute_zero]
        return frozen  # get_temp already auto-evicts; list captures what was purged.

    def _get_current_state(self, domain_id: str) -> dict:
        """
        Reads current decayed temp and writes it back to self.domains
        before any friction is applied. Prevents double-decay on subsequent
        resonate() calls that read stale last_seen timestamps.

        Guard: if get_temp() auto-evicted the key (sublimation), we must NOT
        write the residual 0.0 back — that would silently resurrect a zombie domain.
        """
        if domain_id not in self.domains:
            return {'temp': self.absolute_zero, 'last_seen': time.time(), 'data': None}

        decayed = self.get_temp(domain_id)

        # Key may have been deleted by get_temp() if it crossed sublimation threshold.
        # Only write-back if the domain is still alive AND decayed > absolute_zero.
        if domain_id in self.domains and decayed > self.absolute_zero:
            self.domains[domain_id]['temp'] = decayed
            self.domains[domain_id]['last_seen'] = time.time()
            return self.domains[domain_id]

        return {'temp': self.absolute_zero, 'last_seen': time.time(), 'data': None}

    def status(self) -> dict:
        """Snapshot of all live domain temperatures (does not trigger decay write-back)."""
        return {
            domain_id: round(self.get_temp(domain_id), 2)
            for domain_id in list(self.domains)
        }
