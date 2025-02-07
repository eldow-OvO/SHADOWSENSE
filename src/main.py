import argparse
import logging
import yaml
from stealth_handler import StealthRequestHandler
from ai_detector import AIVulnerabilityDetector
from fuzzing_engine import FuzzingEngine
from reporting import ReportGenerator

def main():
    # Setup
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    stealth = StealthRequestHandler(proxy_list=config["proxies"])
    fuzzer = FuzzingEngine(config["payload_dir"])
    reporter = ReportGenerator()
    
    try:
        ai_model = AIVulnerabilityDetector.load_model("models/shadow_sense_ai")
    except:
        logging.warning("AI model not found! Using placeholder predictions.")
        ai_model = None

    # Run scan
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url", required=True)
    parser.add_argument("-f", "--field", required=True)
    args = parser.parse_args()

    findings = fuzzer.fuzz(args.url, args.field, stealth)
    
    # Analyze results
    for finding in findings:
        if ai_model:
            finding["confidence"] = ai_model.predict(finding["payload"])
        reporter.add_finding(finding)
    
    # Generate report
    print(reporter.generate_report())

if __name__ == "__main__":
    main()
