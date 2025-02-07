import json
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

class FuzzingEngine:
    def __init__(self, payload_dir: str = "data/payloads"):
        self.payloads = self._load_payloads(payload_dir)

    def _load_payloads(self, payload_dir: str) -> Dict[str, List[str]]:
        payloads = {}
        for payload_file in Path(payload_dir).glob("*.json"):
            with open(payload_file) as f:
                payloads[payload_file.stem] = json.load(f)
        return payloads

    def fuzz(self, url: str, field: str, handler) -> List[Dict]:
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for vuln_type, payloads in self.payloads.items():
                for payload in payloads:
                    futures.append(executor.submit(
                        self._test_payload, url, field, payload, vuln_type, handler
                    ))
            for future in futures:
                result = future.result()
                if result:
                    results.append(result)
        return results

    def _test_payload(self, url: str, field: str, payload: str, vuln_type: str, handler):
        response = handler.send_request(url, params={field: payload})
        if response and ("error" in response.text.lower() or "syntax" in response.text.lower()):
            return {"type": vuln_type, "payload": payload, "status": response.status_code}
        return None
