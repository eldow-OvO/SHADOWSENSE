import pytest
from src.ai_detector import AIVulnerabilityDetector

def test_ai_detector():
    detector = AIVulnerabilityDetector()
    X = ["safe input", "malicious input"]
    y = [0, 1]  # 0 = safe, 1 = malicious
    detector.train(X, y)
    assert detector.predict("malicious input") == 1
