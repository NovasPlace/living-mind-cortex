import pytest
from fastapi.testclient import TestClient
from api.main import app
from core.runtime import runtime

@pytest.fixture
def client():
    # Mock the pulse loop so it doesn't cause CancelledError during TestClient teardown
    original_pulse = runtime._pulse_loop
    async def mock_pulse(): pass
    runtime._pulse_loop = mock_pulse
    
    with TestClient(app) as c:
        yield c
        
    runtime._pulse_loop = original_pulse

def test_tier1_happy_path(client: TestClient):
    """Tier 1: Happy path - verify /status and /hormones return valid data."""
    response = client.get("/status")
    assert response.status_code == 200
    
    response = client.get("/hormones")
    assert response.status_code == 200
    assert "dopamine" in response.json()

def test_tier2_error_path(client: TestClient):
    """Tier 2: Error path - endpoints with missing parameters."""
    response = client.get("/memory/recall") # missing 'q'
    assert response.status_code == 422 # Unprocessable Entity

def test_tier3_edge_case(client: TestClient):
    """Tier 3: Edge case - high limits on recall."""
    response = client.get("/memory/recall?q=test&limit=100")
    # pydantic limit is 50, so this should 422
    assert response.status_code == 422

def test_tier4_adversarial(client: TestClient):
    """Tier 4: Adversarial - SQLi attempt in q"""
    response = client.get("/memory/recall?q='; DROP TABLE cortex;--")
    # It shouldn't crash, it should return 200
    assert response.status_code == 200

def test_tier5_telemetry(client: TestClient):
    """Tier 5: Telemetry - checking the stats endpoint."""
    response = client.get("/memory/stats")
    assert response.status_code == 200
    assert "total" in response.json()
