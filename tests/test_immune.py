import pytest
import time
from core.security_perimeter import SecurityPerimeterSystem, QUARANTINE_THRESHOLD

@pytest.fixture
def immune():
    return SecurityPerimeterSystem()

def test_tier1_happy_path(immune):
    """Tier 1: Happy path - report success raises/maintains health."""
    immune.register("organ_a")
    res = immune.report("organ_a", success=True)
    assert res["status"] == "healthy"
    assert res["health"] == 100.0

def test_tier2_error_path(immune):
    """Tier 2: Error path - automatic registration on report failure."""
    res = immune.report("unregistered_organ", success=False)
    assert res["status"] == "healthy" # only drops 20, still healthy
    assert res["health"] == 80.0

def test_tier3_edge_case(immune):
    """Tier 3: Edge case - rapid failures trigger quarantine."""
    immune.register("organ_crash")
    for _ in range(QUARANTINE_THRESHOLD):
        immune.report("organ_crash", success=False)
    
    assert immune.is_quarantined("organ_crash")

def test_tier4_adversarial(immune):
    """Tier 4: Adversarial - check rapid fire rate limiting."""
    immune.register("spam_organ")
    # Fire quickly
    for _ in range(6):
        immune.report("spam_organ", success=True)
        # We manually overwrite the fire_times to simulate instant firing
    
    assert immune.is_rate_limited("spam_organ")

@pytest.mark.asyncio
async def test_tier5_telemetry(immune):
    """Tier 5: Telemetry - check patrol report and inflammation."""
    immune.report("sick", success=False)
    immune.report("sick", success=False)
    immune.report("sick", success=False) # quarantined
    
    patrol_res = await immune.patrol(pulse=1)
    assert patrol_res["quarantined"] == 1
    assert patrol_res["healthy"] == 0
    assert patrol_res["inflammation"] == 1.0 # 1 out of 1 sick
