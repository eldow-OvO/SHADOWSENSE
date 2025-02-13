import json
import os
from concurrent.futures import ThreadPoolExecutor

class FuzzingEngine:
    def __init__(self, payload_dir="data/payloads"):
        self.payloads = self._load_payloads(payload_dir)
        
    def _load_payloads(self, payload_dir):
        payloads = {}
        for file in os.listdir(payload_dir):
            if file.endswith(".json"):
                with open(os.path.join(payload_dir, file)) as f:
                    payloads[file.split(".")[0]] = json.load(f)
        return payloads

    def fuzz(self, url, param, handler):
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for vuln_type in self.payloads:
                for payload in self.payloads[vuln_type]:
                    futures.append(executor.submit(
                        self._test_payload, url, param, payload, vuln_type, handler
                    ))
            for future in futures:
                result = future.result()
                if result:
                    results.append(result)
        return results

    def _test_payload(self, url, param, payload, vuln_type, handler):
        response = handler.send_request(url, {param: payload})
        if response and ("error" in response.text.lower() or "syntax" in response.text.lower()):
            return {"type": vuln_type, "payload": payload, "status": response.status_code}
        return None
