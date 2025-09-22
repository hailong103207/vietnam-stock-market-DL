import requests
import json
import pandas as pd
import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIHandler:
    BASE_URL: str = "https://api.vietstock.vn/tvnew/history"
    HEADERS: Dict[str, str] = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Origin": "https://stockchart.vietstock.vn",
        "Referer": "https://stockchart.vietstock.vn/"
    }

    def __init__(self, timeout: int = 5, retries: int = 3, backoff_factor: float = 0.3):
        self.timeout = timeout
        self.session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def fetch_history(
        self,
        from_ts: int,
        to_ts: int,
        ticker: str,
        resolution: str
    ) -> Optional[Dict[str, Any]]:
        """
        Lấy dữ liệu lịch sử của cổ phiếu từ Vietstock API.
        """
        
        params = {
            "symbol": ticker,
            "resolution": resolution,
            "from": from_ts,
            "to": to_ts,
            "countback": 2
        }

        try:
            response = self.session.get(
                self.BASE_URL,
                headers=self.HEADERS,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            #add params to response
            data = response.json()
            data["params"] = params
            #save json to file
            with open(f"data/json/cache.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        
    def fetch_realtime_data(self,resolution, range, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Lấy dữ liệu thời gian thực của cổ phiếu từ Vietstock API.
        """
        to_date = pd.Timestamp.now()
        from_date = to_date - pd.Timedelta(range)
        return self.fetch_history(from_date, to_date, ticker, resolution)
    


if __name__ == "__main__":
    APIHandler.fetch_realtime_data("1", "1D", "VND")