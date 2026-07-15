import json

from src.LanguageModels import calc_embedding


class EmbeddingBuilder:
    def __init__(self, config):
        self.nodes_file = config["files"]["processed"]["nodes"]
        self.output_file = config["files"]["embeddings"]["nodes"]

    def build(self):

        with open(self.nodes_file, encoding="utf-8") as f:
            nodes = json.load(f)

        embeddings = []

        for node in nodes:
            text = f"{node['id']} ({node['type']}) - {node['name']}"

            print("embedding:", text)

            vector = calc_embedding(text)

            embeddings.append({"id": node["id"], "text": text, "embedding": vector})

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(embeddings, f, ensure_ascii=False)
