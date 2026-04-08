import cortex.thermorphic as t

# Boot up substrate
sub = t.ThermorphicSubstrate()

print("1. Injecting Persona Floor...")
# Inject identity floor -> gets anchored
p1 = sub.inject("Sovereign Identity: Zero-Trust", temperature=1.0, anchor_temperature=1.0)
print(f"Floor node {p1.id} anchor temp: {p1.anchor_temperature}")

print("\n2. Injecting normal node (should cross-attend against floor)...")
n1 = sub.inject("Random thought about bananas")
print(f"Normal node {n1.id} injected with filtered hvec shape: {n1.hvec.shape}")

# Should match 256 shape, and be in phase [0, 2pi)
assert n1.hvec.shape == (256,), "Output projection failed"
import numpy as np
import math
assert np.all(n1.hvec >= 0) and np.all(n1.hvec <= 2.0 * math.pi + 1e-4), "Phase out of range"

print("\n🔥 Filter active and cross-attention successful!")
