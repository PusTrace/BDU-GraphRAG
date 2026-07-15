## build:

- DataScrapper.py - get files from api
- Parser.py - parse files from raw to nodes.json and edges.json
- NodesVectors.py - for calc embedings for vector search
- FAISS_index.py = for indexing nodes

## start:

- CUDA_VISIBLE_DEVICES="" ./build/bin/llama-server -m models/Qwen3-Embedding-0.6B-Q8_0.gguf --embeddings --port 8081
- ./build/bin/llama-server -m models/Qwen3-VL-4B-Instruct-Q5_K_M.gguf --port 8080 -ngl 99
- api.py - user view (available only console view)

## other:

- GraphSearch.py - module for search in graph and return node + neighbor nodes
- config.json - for configuration scripts
- llm.py - for chat with llm

~ mean file not did but planned
