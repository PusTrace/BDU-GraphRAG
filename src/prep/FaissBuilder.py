import json
import numpy as np
import faiss


class FaissBuilder:
    def __init__(self, config):
        self.embeddings_file = config["files"]["embeddings"]["nodes"]
        self.index_file = config["files"]["index"]["nodes"]

    def build(self):

        with open(self.embeddings_file, encoding="utf-8") as f:
            data = json.load(f)

        vectors = np.array([item["embedding"] for item in data], dtype="float32")

        dimension = vectors.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(vectors)

        faiss.write_index(index, self.index_file)

        print("FAISS:", index.ntotal)
