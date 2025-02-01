import pytest
from src.stealth_handler import StealthRequestHandler

def test_stealth_handler():
    handler = StealthRequestHandler()
    response = handler.send_request("https://example.com")
    assert response is not None
    assert response.status_code == 200
