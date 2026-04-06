import pytest
from state.telemetry_broker import TelemetryBroker, BASELINES

@pytest.fixture
def bus():
    return TelemetryBroker()

def test_tier1_happy_path(bus):
    """Tier 1: Happy path - injecting emotion properly shifts hormones."""
    bus.inject_emotion("joy")
    assert bus.state.dopamine > BASELINES["dopamine"]
    assert bus.state.serotonin > BASELINES["serotonin"]
    assert bus.state.cortisol < BASELINES["cortisol"]

def test_tier2_error_path(bus):
    """Tier 2: Error path - invalid hormone injection is ignored safely."""
    # Should not raise an error
    bus.inject("invalid_hormone", 1.0)
    # the state should not have this attribute
    assert not hasattr(bus.state, "invalid_hormone")

def test_tier3_edge_case(bus):
    """Tier 3: Edge case - hormone clipping behavior bounds check."""
    bus.inject("dopamine", 10.0)
    assert bus.state.dopamine == 1.0

    bus.inject("dopamine", -20.0)
    assert bus.state.dopamine == 0.0

def test_tier4_adversarial(bus):
    """Tier 4: Adversarial - checking for type safety/fault tolerance"""
    # Assuming the implementation might crack under weird types; 
    # Python will throw TypeError since it's doing math, but it shouldn't corrupt the rest
    with pytest.raises(TypeError):
        bus.inject("dopamine", "not-a-float")

def test_tier5_telemetry(bus):
    """Tier 5: Telemetry - checking if events are properly logged."""
    assert len(bus._event_log) == 0
    bus.inject("dopamine", 0.1, source="test")
    assert len(bus._event_log) == 1
    assert bus._event_log[-1]["source"] == "test"
    
    # Check capping at 50
    for i in range(60):
        bus.inject("dopamine", 0.01)
    
    assert len(bus._event_log) == 50
