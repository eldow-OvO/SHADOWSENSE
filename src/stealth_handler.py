import requests
import random
import time
from typing import Optional, Dict, Union
from fake_useragent import UserAgent
import logging

class StealthRequestHandler:
    """Handles HTTP requests with stealth techniques to avoid detection."""
    
    def __init__(self, max_retries: int = 3, proxy_list: Optional[list] = None):
        self.user_agent = UserAgent()
        self.proxies = proxy_list or []
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

    def _get_random_headers(self) -> Dict[str, str]:
        """Generate random headers to mimic a real browser."""
        return {
            "User-Agent": self.user_agent.random,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }

    def send_request(
        self, 
        url: str, 
        method: str = "GET", 
        params: Optional[Dict] = None, 
        data: Optional[Dict] = None
    ) -> Optional[requests.Response]:
        """Send a stealthy HTTP request with retries and proxy rotation."""
        headers = self._get_random_headers()
        
        for attempt in range(self.max_retries):
            try:
                proxy = {"http": random.choice(self.proxies), "https": random.choice(self.proxies)} if self.proxies else None
                response = requests.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    headers=headers,
                    proxies=proxy,
                    timeout=10,
                    verify=False  # Disable SSL verification (use cautiously)
                )
                
                # Random delay to mimic human behavior
                time.sleep(random.uniform(1, 5))
                
                if response.status_code == 200:
                    return response
                else:
                    self.logger.warning(f"Request failed (Status {response.status_code}). Retrying...")
            
            except (requests.RequestException, ConnectionError) as e:
                self.logger.error(f"Request error: {e}. Attempt {attempt + 1}/{self.max_retries}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
