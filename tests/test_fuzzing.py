import pytest
from src.fuzzing_engine import FuzzingEngine

def test_fuzzing_engine():
    fuzzer = FuzzingEngine()
    assert len(fuzzer.payloads) > 0
