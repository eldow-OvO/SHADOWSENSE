from stealth_handler import StealthRequestHandler
from ai_detector import AIVulnerabilityDetector
from fuzzing_engine import FuzzingEngine
from reporting import ReportingModule

if __name__ == "__main__":
    # Initialize modules
    stealth_handler = StealthRequestHandler()
    ai_detector = AIVulnerabilityDetector()
    fuzzing_engine = FuzzingEngine()
    reporter = ReportingModule()

    # Target URL and input field
    target_url = "http://example.com/search"
    input_field = "query"

    # Fuzz the target
    results = fuzzing_engine.fuzz(target_url, input_field)

    # Analyze results with AI
    for payload, _ in results:
        ai_prediction = ai_detector.predict(payload)
        if ai_prediction == 1:  # 1 indicates vulnerability
            reporter.add_finding("SQL Injection", payload, target_url)

    # Generate report
    reporter.generate_report()
