import argparse
import logging
import yaml
from stealth_handler import StealthRequestHandler
from ai_detector import AIVulnerabilityDetector
from fuzzing_engine import FuzzingEngine
from reporting import ReportGenerator

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    stealth = StealthRequestHandler(proxies=config.get("proxies", []))
    fuzzer = FuzzingEngine(config.get("payload_dir", "data/payloads"))
    reporter = ReportGenerator()
    
    # Load AI model
    try:
        ai_model = AIVulnerabilityDetector.load_model("models/shadow_sense_ai")
    except:
        logging.warning("AI model not found! Running without AI validation.")
        ai_model = None

    # Parse arguments
    parser = argparse.ArgumentParser(description="ShadowSense Vulnerability Scanner")
    parser.add_argument("-u", "--url", required=True, help="Target URL to scan")
    parser.add_argument("-p", "--param", required=True, help="Parameter to test")
    args = parser.parse_args()

    # Run scan
    findings = fuzzer.fuzz(args.url, args.param, stealth)
    
    # Analyze findings
    for finding in findings:
        if ai_model:
            confidence = ai_model.predict(finding["payload"])
            finding["confidence"] = round(confidence, 2)
        reporter.add_finding(finding)
    
    # Generate report
    print(reporter.generate_report())

if __name__ == "__main__":
    main()
