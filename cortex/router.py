import numpy as np
import json
from cortex.heatsink import ThermomorphicMemoryPlasma

class BiomechanicRouter:
    """
    Determines LoRA adapter state via thermal resonance of the HSM.

    Routing is a two-stage process:
      1. Geometric Stage  — cosine/distance resonance against Cortex DB vectors.
      2. Thermal Stage    — plasma temperature from the CognitiveHeatsink scales
                           each domain's accumulated score. Hot domains get up to
                           a 2x multiplier; sublimated domains (temp=0) are
                           completely blocked from routing.

    After every successful expert route, the winning domain is resonated in the
    heatsink, ensuring repeated use keeps the LoRA adapter warm in VRAM.
    """
    def __init__(self, cortex_engine, heatsink: ThermomorphicMemoryPlasma = None):
        self.cortex = cortex_engine
        # Inject or create a default heatsink. A shared instance should be passed
        # in from the application lifespan so the API gateway and router share
        # the same plasma state.
        self.heatsink = heatsink or ThermomorphicMemoryPlasma(cooling_constant=0.005)
        # Define the threshold where an expert overrides the general base model
        self.activation_threshold = 0.75

    async def route_prompt(self, prompt_hvec: np.ndarray) -> str:
        """
        Sweeps the incoming prompt against the substrate. 
        Returns the required LoRA adapter ID dynamically based on thermal aggregation.
        """
        # 1. Fire the dual-stage HSM sweep
        resonating_node_ids = await self.cortex.find_resonating_nodes(
            prompt_hvec, 
            threshold=0.75
        )
        
        if not resonating_node_ids:
            return "base_model"
            
        # 2. Fetch the full memory nodes from Cortex DB to extract metadata and importances
        # We need the semantic_vector (embedding) and importance
        nodes = []
        async with self.cortex._pool.acquire() as conn:
            query = """
                SELECT id, importance, metadata, embedding
                FROM memories 
                WHERE id = ANY($1::uuid[])
            """
            rows = await conn.fetch(query, resonating_node_ids)
            for row in rows:
                metadata = row['metadata']
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                else:
                    metadata = metadata or {}
                    
                nodes.append({
                    "id": str(row["id"]),
                    "importance": row["importance"],
                    "metadata": metadata,
                    "semantic_vector": np.frombuffer(row["embedding"], dtype=np.float32) if row["embedding"] else None
                })
        
        # 3. Aggregate geometric thermal mass by domain
        domain_heat = {"code_expert": 0.0, "logic_expert": 0.0, "base_model": 0.0}

        for node in nodes:
            if node["semantic_vector"] is None:
                continue

            domain = node["metadata"].get("cognitive_domain", "base_model")

            # Geometric score: importance weighted by proximity resonance
            distance = np.linalg.norm(prompt_hvec - node["semantic_vector"])
            resonance_multiplier = 1.0 / (distance + 0.001)

            if domain not in domain_heat:
                domain_heat[domain] = 0.0

            domain_heat[domain] += (node['importance'] * resonance_multiplier)

        # 4. Apply Thermal Stage — scale each domain's score by its plasma temperature.
        #    Sublimated domains (temp == 0) are hard-blocked: they cannot win the route
        #    because their LoRA has been evicted from VRAM.
        plasma_status = self.heatsink.status()
        for domain in list(domain_heat.keys()):
            if domain == "base_model":
                continue  # Base substrate is always available; not managed by heatsink.

            plasma_temp = self.heatsink.get_temp(domain)

            if plasma_temp == self.heatsink.absolute_zero:
                # Domain has sublimated — LoRA is not in VRAM. Hard-block it.
                domain_heat[domain] = 0.0
                print(f"[MoERouter] Domain '{domain}' is sublimated. Blocking route.")
            else:
                # Thermal amplification: 1.0x (cold) → 2.0x (maxed at 500K)
                thermal_weight = 1.0 + (plasma_temp / 500.0)
                domain_heat[domain] *= thermal_weight

        print(f"[MoERouter] Thermal Aggregation (post-plasma): {{"
              + ", ".join(f"'{k}': {v:.3f}" for k, v in domain_heat.items())
              + f"}} | Plasma: {plasma_status}")

        # 5. Determine the dominant hemisphere
        dominant_domain = max(domain_heat, key=domain_heat.get)
        peak_heat = domain_heat[dominant_domain]

        if dominant_domain != "base_model" and peak_heat >= self.activation_threshold:
            # Heat up the winning expert — keeps it alive in VRAM for the next call.
            new_temp = self.heatsink.resonate(dominant_domain)
            print(f"[MoERouter] Threshold breached. Routing to '{dominant_domain}' "
                  f"(plasma now {new_temp:.1f}K).")
            return dominant_domain

        print("[MoERouter] Operating in Base Model (Orchestration) mode.")
        return "base_model"
