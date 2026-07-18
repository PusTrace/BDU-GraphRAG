# collect and save data in database
from urllib.parse import urlencode
import requests
import json


class BDU_API:
    def __init__(self, bearer_token: str, phpsessid: str, csrf: str, config):
        self.host = "https://bdu.fstec.ru"
        self.API_CONFIG = config["api"]
        self.files = config["files"]["raw"]
        self.session = requests.Session()

        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
                "Referer": self.host,
                "Authorization": f"Bearer {bearer_token}",
            }
        )

        self.session.cookies.update(
            {
                "PHPSESSID": phpsessid,
                "YII_CSRF_TOKEN": csrf,
            }
        )
        self.session.verify = False  # TODO: fix cert in system

    def build_url(self, path: str, params: dict | None = None) -> str:
        url = f"{self.host}{path}"
        if params:
            url += "?" + urlencode(params, doseq=True)
        return url

    def get(self, path: str, params: dict | None = None):
        url = self.build_url(path, params)
        return self.session.get(url)

    def fetch_list(self, target: str):
        config = self.API_CONFIG[target]

        params = {
            "page[limit]": 1000,
            "page[offset]": 0,
        }

        if config["page_include"]:
            params["include"] = ",".join(config["page_include"])

        resp = self.get(config["endpoint"], params)
        resp.raise_for_status()

        data = resp.json()

        file_path = self.files[target]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved {file_path}")

        return data

    def fetch_all_lists(self):
        for target in self.API_CONFIG:
            try:
                self.fetch_list(target)
            except Exception as e:
                print(f"[ERROR] {target}: {e}")

    def fetch_target_list(self, target: str):
        if target not in self.API_CONFIG:
            raise ValueError(f"Unknown target: {target}")

        return self.fetch_list(target)

    def fetch_item(self, target: str, identifier: str):
        config = self.API_CONFIG[target]

        params = {}

        if config["item_include"]:
            params["include"] = ",".join(config["item_include"])

        endpoint = f"{config['endpoint']}/{identifier}"

        resp = self.get(endpoint, params)
        resp.raise_for_status()

        return resp.json()
