import os
from dotenv import load_dotenv
import json

import src.storage as load
from src.prep.DataScrapper import BDU_API
from src.prep.Parser import BDUParser
from src.prep.EmbeddingBuilder import EmbeddingBuilder
from src.prep.FaissBuilder import FaissBuilder


def main():
    print("data_scrapper")
    load_dotenv()
    config = load.config()
    api = BDU_API(
        bearer_token=os.getenv("bearer_token"),
        phpsessid=os.getenv("phpsessid"),
        csrf=os.getenv("csrf"),
        config=config,
    )
    api.fetch_all_lists()

    print("Parser")
    config = load.config()
    parser = BDUParser(config=config)
    files = config["files"]["processed"]

    nodes, edges = parser.run()
    with open(files["nodes"], "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)

    with open(files["edges"], "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)

    print("EmbeddingBuilder")
    EmbeddingBuilder(config).build()
    print("FaissBuilder")
    FaissBuilder(config).build()


if __name__ == "__main__":
    config = load.config()
    FaissBuilder(config).build()
    # main()
