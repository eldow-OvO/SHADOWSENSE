from datetime import datetime
import json

class ReportGenerator:
    def __init__(self):
        self.findings = []

    def add_finding(self, finding: Dict):
        self.findings.append(finding)

    def generate_report(self, format: str = "console") -> str:
        if format == "json":
            return json.dumps(self.findings, indent=2)
        elif format == "html":
            return self._generate_html()
        else:
            report = [f"[{f['type']}] {f['payload']} (Status: {f['status']})" for f in self.findings]
            return "\n".join(report)

    def _generate_html(self) -> str:
        html = f"<h1>ShadowSense Report - {datetime.now()}</h1><ul>"
        for finding in self.findings:
            html += f"<li><strong>{finding['type']}</strong>: {finding['payload']} (Status: {finding['status']})</li>"
        return html + "</ul>"
