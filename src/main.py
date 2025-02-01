import argparse
import logging
import yaml
from pathlib import Path
from stealth_handler import StealthRequestHandler
from ai_detector import AIVulnerabilityDetector
from fuzzing_engine import FuzzingEngine
from reporting import ReportGenerator

def configure_logging(log_level: str = "INFO") -> None:
    """Configure logging format and level."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("shadow_sense.log"),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="ShadowSense: AI-Powered Web Vulnerability Scanner")
    parser.add_argument("-u", "--url", required=True, help="Target URL to scan")
    parser.add_argument("-f", "--field", required=True, help="Input field to test")
    parser.add_argument("-c", "--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("-o", "--output", choices=["console", "json", "html", "markdown"], default="console", help="Report format")
    args = parser.parse_args()

    # Initialize components
    configure_logging()
    config = load_config(args.config)
    
    stealth_handler = StealthRequestHandler(
        max_retries=config.get("max_retries", 3),
        proxy_list=config.get("proxies", [])
    )
    
    fuzzer = FuzzingEngine(config.get("payload_dir", "data/payloads"))
    reporter = ReportGenerator()
    
    # Load or train AI model
    ai_model_path = config.get("ai_model_path", "models/shadow_sense_ai")
    if Path(f"{ai_model_path}_model.pth").exists():
        ai_detector = AIVulnerabilityDetector.load_model(ai_model_path)
    else:
        ai_detector = AIVulnerabilityDetector(input_dim=1000)
        # TODO: Load training data and train
        # ai_detector.train(X_train, y_train)
        # ai_detector.save_model(ai_model_path)

    # Run scan
    findings = fuzzer.fuzz(args.url, args.field, stealth_handler)
    
    # Analyze findings with AI
    for finding in findings:
        probability = ai_detector.predict(finding["payload"])
        if probability > 0.7:  # Threshold
            reporter.add_finding(finding)
    
    # Generate report
    report = reporter.generate_report(args.output)
    if args.output != "console" and report:
        with open(f"report_{datetime.now().strftime('%Y%m%d')}.{args.output}", "w") as f:
            f.write(report)

if __name__ == "__main__":
    main()
