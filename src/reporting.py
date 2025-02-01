from typing import List, Dict
import json
import html
from datetime import datetime
import logging

class ReportGenerator:
    """Generates vulnerability reports in multiple formats."""
    
    def __init__(self):
        self.reports = []
        self.logger = logging.getLogger(__name__)
    
    def add_finding(self, finding: Dict) -> None:
        """Add a vulnerability finding to the report."""
        self.reports.append(finding)
        self.logger.info(f"Added finding: {finding['type']}")
    
    def generate_report(self, format: str = "console") -> Optional[str]:
        """Generate report in specified format (console/json/html/markdown)."""
        if not self.reports:
            return None
        
        if format == "json":
            return json.dumps(self.reports, indent=2)
        
        elif format == "html":
            return self._generate_html()
        
        elif format == "markdown":
            return self._generate_markdown()
        
        else:  # Default to console
            for report in self.reports:
                print(f"[{report['type']}] Payload: {report['payload']}")
                print(f"Response Time: {report['response_time']}s | Status: {report['status_code']}")
                print("-" * 50)
            return None
    
    def _generate_html(self) -> str:
        """Generate HTML report."""
        html_content = """
        <html>
        <head><title>Vulnerability Report</title></head>
        <body>
            <h1>ShadowSense Report - {date}</h1>
            <table border="1">
                <tr>
                    <th>Type</th>
                    <th>Payload</th>
                    <th>Status Code</th>
                    <th>Response Time</th>
                </tr>
        """.format(date=datetime.now().strftime("%Y-%m-%d"))
        
        for report in self.reports:
            html_content += f"""
                <tr>
                    <td>{html.escape(report['type'])}</td>
                    <td><code>{html.escape(report['payload'])}</code></td>
                    <td>{report['status_code']}</td>
                    <td>{report['response_time']}s</td>
                </tr>
            """
        
        html_content += "</table></body></html>"
        return html_content
    
    def _generate_markdown(self) -> str:
        """Generate Markdown report."""
        md_content = f"# ShadowSense Report - {datetime.now().strftime('%Y-%m-%d')}\n\n"
        md_content += "| Type | Payload | Status Code | Response Time |\n"
        md_content += "|------|---------|-------------|---------------|\n"
        
        for report in self.reports:
            md_content += f"| {report['type']} | `{report['payload']}` | {report['status_code']} | {report['response_time']}s |\n"
        
        return md_content
