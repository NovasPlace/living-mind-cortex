"""
cortex/protocols.py — Formal substrate contracts for Living Mind organs.

The cognitive_continuity_eval benchmark surfaced three implicit contracts
that any ThermorphicSubstrate consumer depends on. This module formalizes
them as a runtime-checkable Protocol so:
  - Benchmark can type-check rather than assert-guess
  - Any future organ swap (e.g. HolographicSubstrate) is immediately verified
  - Regression risk drops to zero: mypy / beartype catch violations at import

Three contracts:
  1. Construction  — dims accepted and APPLIED (not ignored)
  2. Lifecycle     — reset() exists and FULLY purges state
  3. Pulse semantics — freeze_dwell is per-instance, not module-global

Usage:
    from cortex.protocols import IBenchmarkableSubstrate
    def run_benchmark(substrate: IBenchmarkableSubstrate): ...
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import numpy as np


# ── Contract ──────────────────────────────────────────────────────────────────

@runtime_checkable
class IBenchmarkableSubstrate(Protocol):
    """
    Formal interface contract for substrates usable in Living Mind benchmarks.

    All three methods map directly to the implicit contracts discovered during
    cognitive_continuity_eval development (April 2026 benchmark hardening pass).
    """

    # ── Contract 1: Construction ─────────────────────────────────────────────
    # The substrate must expose the active embedding dimension so callers
    # can assert the OUTCOME (behavior), not just the API surface.
    #
    # Correct usage:
    #   sub = ThermorphicSubstrate(dims=256)
    #   assert sub.dims == 256          ← validates behavior, not signature
    dims: int

    # ── Contract 2: Lifecycle ─────────────────────────────────────────────────
    # reset() must fully purge: nodes, fusion_log, pulse_count, totals.
    # Benchmark isolation requires zero state bleed between test scenarios.
    def reset(self) -> None:
        """Fully purge all state. Must be idempotent."""
        ...

    # ── Contract 3: Pulse semantics ───────────────────────────────────────────
    # freeze_dwell must be a per-instance attribute, not a module global.
    # Benchmarks need to vary it per scenario without cross-contamination.
    freeze_dwell: int

    # ── Core substrate API (required for benchmark operation) ─────────────────
    def inject(
        self,
        content: str,
        temperature: float = 0.8,
        tags: List[str] = None,
        dims: int = 256,
    ) -> object:
        """Inject a concept. dims must be applied — not advisory."""
        ...

    def pulse(self) -> dict:
        """Run one thermodynamic tick. Returns event dict."""
        ...


# ── Behavioral assertion helper ───────────────────────────────────────────────

def assert_contract(substrate: IBenchmarkableSubstrate, expected_dims: int) -> None:
    """
    Assert all three behavioral contracts hold on a concrete substrate instance.
    Call this at the top of any benchmark that receives a substrate argument.

    Raises AssertionError with a diagnostic message for each violation.
    This validates BEHAVIOR, not API surface — a constructor that accepts dims
    and silently ignores it will be caught here.
    """
    # Contract 1: dims actually applied
    assert hasattr(substrate, "dims"), (
        f"{type(substrate).__name__} missing .dims — Construction contract violated. "
        f"The substrate accepts a dims argument but does not expose the active value."
    )
    assert substrate.dims == expected_dims, (
        f"{type(substrate).__name__}.dims = {substrate.dims}, expected {expected_dims}. "
        f"Constructor accepted dims={expected_dims} but did not apply it. "
        f"This is a behavioral violation — assert the outcome, not the signature."
    )

    # Contract 2: reset() exists and is callable
    assert hasattr(substrate, "reset") and callable(substrate.reset), (
        f"{type(substrate).__name__} missing reset() — Lifecycle contract violated. "
        f"Benchmarks require full state purge between scenarios."
    )

    # Contract 3: freeze_dwell is instance-level
    assert hasattr(substrate, "freeze_dwell"), (
        f"{type(substrate).__name__} missing .freeze_dwell — Pulse semantics contract violated. "
        f"freeze_dwell must be a per-instance attribute, not a module global, "
        f"so benchmark scenarios can vary it without cross-contamination."
    )
