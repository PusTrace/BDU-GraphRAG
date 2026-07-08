import json
import numpy as np
import faiss


with open("data/node_embeddings.json", encoding="utf-8") as f:
    data = json.load(f)


vectors = []

for item in data:
    vectors.append(item["embedding"])


vectors = np.array(vectors, dtype="float32")


print("vectors:", vectors.shape)

dimension = vectors.shape[1]


index = faiss.IndexFlatL2(dimension)


index.add(vectors)


print("index size:", index.ntotal)

faiss.write_index(index, "data/nodes.index")
