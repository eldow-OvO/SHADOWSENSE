class ReportGenerator:
    def __init__(self):
        self.findings = []
        
    def add_finding(self, finding):
        self.findings.append(finding)
        
    def generate_report(self):
        if not self.findings:
            return "No vulnerabilities found!"
            
        report = ["Vulnerability Report:"]
        for idx, finding in enumerate(self.findings, 1):
            report.append(
                f"{idx}. Type: {finding['type']}\n"
                f"   Payload: {finding['payload']}\n"
                f"   Status Code: {finding['status']}"
            )
        return "\n".join(report)
