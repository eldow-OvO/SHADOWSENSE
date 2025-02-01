import json
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

class FuzzingEngine:
    """Advanced fuzzing engine with parallel requests and payload management."""
    
    def __init__(self, payload_dir: str = "data/payloads"):
        self.payloads = self._load_payloads(payload_dir)
        self.logger = logging.getLogger(__name__)
    
    def _load_payloads(self, payload_dir: str) -> Dict[str, List[str]]:
        """Load payloads from JSON files categorized by vulnerability type."""
        payloads = {}
        payload_dir = Path(payload_dir)
        
        for payload_file in payload_dir.glob("*.json"):
            with open(payload_file, "r") as f:
                payloads[payload_file.stem] = json.load(f)
        
        return payloads
    
    def fuzz(
        self, 
        url: str, 
        input_field: str, 
        stealth_handler: "StealthRequestHandler",
        max_workers: int = 5
    ) -> List[Dict[str, Union[str, float]]]:
        """Run parallel fuzzing attacks with categorized payloads."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for vuln_type, payload_list in self.payloads.items():
                for payload in payload_list:
                    futures.append(
                        executor.submit(
                            self._test_payload,
                            url,
                            input_field,
                            payload,
                            vuln_type,
                            stealth_handler
                        )
                    )
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        
        return results
    
    def _test_payload(
        self,
        url: str,
        input_field: str,
        payload: str,
        vuln_type: str,
        stealth_handler: "StealthRequestHandler"
    ) -> Optional[Dict[str, Union[str, float]]]:
        """Test a single payload and return results."""
        try:
            response = stealth_handler.send_request(
                url,
                method="POST",
                data={input_field: payload}
            )
            
            if response and self._is_vulnerable(response):
                return {
                    "type": vuln_type,
                    "payload": payload,
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                }
        
        except Exception as e:
            self.logger.error(f"Failed to test payload {payload}: {e}")
        
        return None
    
    def _is_vulnerable(self, response) -> bool:
        """Heuristic check for vulnerability indicators."""
        indicators = [
            "error in your SQL syntax",
            "warning: mysql",
            "unexpected token",
            "root:",
            "exception"
        ]
        return any(indicator in response.text.lower() for indicator in indicators)
