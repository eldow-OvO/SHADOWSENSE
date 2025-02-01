import requests
import random
import time
import logging
from typing import Optional, Dict, List

class StealthRequestHandler:
    def __init__(self, max_retries: int = 3, proxy_list: Optional[List[str]] = None):
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        self.proxies = proxy_list or []
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

    def _get_random_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    def send_request(self, url: str, method: str = "GET", params: Optional[Dict] = None) -> Optional[requests.Response]:
        for attempt in range(self.max_retries):
            try:
                proxy = {"http": random.choice(self.proxies)} if self.proxies else None
                response = requests.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=self._get_random_headers(),
                    proxies=proxy,
                    timeout=10,
                    verify=False
                )
                time.sleep(random.uniform(1, 3))
                return response
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
        return None
