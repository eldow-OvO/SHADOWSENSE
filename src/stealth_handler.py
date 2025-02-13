import requests
import random
import time
import logging

class StealthRequestHandler:
    def __init__(self, max_retries=3, proxies=None):
        self.proxies = proxies or []
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
        
    def _get_headers(self):
        return {
            "User-Agent": random.choice([
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15"
            ]),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }

    def send_request(self, url, params=None):
        for attempt in range(self.max_retries):
            try:
                proxy = {"http": random.choice(self.proxies)} if self.proxies else None
                response = requests.get(
                    url,
                    params=params,
                    headers=self._get_headers(),
                    proxies=proxy,
                    timeout=10,
                    verify=False
                )
                time.sleep(random.uniform(1, 3))
                return response
            except Exception as e:
                self.logger.error(f"Attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        return None
