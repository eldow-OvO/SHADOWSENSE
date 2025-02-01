import requests
import random
import time
from fake_useragent import UserAgent

class StealthRequestHandler:
    def __init__(self):
        self.user_agent = UserAgent()
        self.proxies = self._load_proxies()  # Load a list of proxies (optional)

    def _load_proxies(self):
        # Load proxies from a file or API
        return ["http://proxy1:port", "http://proxy2:port"]

    def _get_random_proxy(self):
        return random.choice(self.proxies) if self.proxies else None

    def send_request(self, url, method="GET", params=None, headers=None):
        try:
            # Randomize headers and proxies
            headers = headers or {}
            headers["User-Agent"] = self.user_agent.random
            proxy = self._get_random_proxy()

            # Send request
            response = requests.request(
                method, url, params=params, headers=headers, proxies={"http": proxy, "https": proxy}
            )
            time.sleep(random.uniform(1, 3))  # Delay to avoid detection
            return response
        except Exception as e:
            print(f"Request failed: {e}")
            return None
