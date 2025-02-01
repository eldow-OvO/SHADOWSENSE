class ReportingModule:
    def __init__(self):
        self.report = []

    def add_finding(self, vulnerability, payload, url):
        self.report.append({
            "url": url,
            "vulnerability": vulnerability,
            "payload": payload
        })

    def generate_report(self):
        for entry in self.report:
            print(f"URL: {entry['url']}")
            print(f"Vulnerability: {entry['vulnerability']}")
            print(f"Payload: {entry['payload']}")
            print("-" * 40)
