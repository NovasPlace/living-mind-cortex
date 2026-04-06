import pytest
from chemistry.circadian import CircadianClock

@pytest.fixture
def clock():
    return CircadianClock()

class DummyBus:
    def __init__(self):
        self.injected = []
    def inject(self, hormone, delta, source):
        self.injected.append((hormone, delta))

@pytest.mark.asyncio
async def test_tier1_happy_path(clock):
    """Tier 1: Happy path - circadian pulse returns snapshot."""
    bus = DummyBus()
    snap = await clock.pulse(1, bus)
    assert "phase" in snap
    assert "adenosine" in snap

def test_tier2_error_path(clock):
    """Tier 2: Error path - weird hours fall back to night?"""
    # Although python's datetime handles 0-23, what if we pass invalid manually just to see internals?
    phase = clock._compute_phase(25) # Invalid hour handling falls to night
    assert phase == "night"

def test_tier3_edge_case(clock):
    """Tier 3: Edge case - high adenosine triggers melatonin spike."""
    clock.adenosine = 0.9
    bus = DummyBus()
    # Mock datetime is hard without lib, let's just run pulse
    import asyncio
    asyncio.run(clock.pulse(1, bus))
    # the DummyBus should have caught the melatonin injection
    melatonin_injections = [x for x in bus.injected if x[0] == "melatonin"]
    assert len(melatonin_injections) > 0

def test_tier4_adversarial(clock):
    """Tier 4: Adversarial - Check adenosine clipping."""
    clock.adenosine = 1.0
    # Add manual increase
    clock.adenosine += 0.5
    # Then next pulse it gets clipped
    bus = DummyBus()
    import asyncio
    asyncio.run(clock.pulse(1, bus))
    assert clock.adenosine <= 1.0

def test_tier5_telemetry(clock):
    """Tier 5: Telemetry - snapshot outputs expected keys without crashing."""
    snap = clock.snapshot()
    assert snap["transitions"] >= 0
    assert isinstance(snap["pulse_scale"], float)
