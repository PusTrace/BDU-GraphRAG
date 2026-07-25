# api for chat with llm
import json

import src.storage as load
from src.Graph import Graph
from src.LanguageModels import (
    create_context,
    slm,
    slm_GraphRAG,
    slm_RAG,
    create_graph_context,
)


def main():
    config = load.config()
    files = config["files"]
    graph = Graph(
        files["processed"]["nodes"],
        files["processed"]["edges"],
        files["index"]["nodes"],
        files["embeddings"]["nodes"],
    )
    try:
        while True:
            # model = input(
            #    "choose model:\n1. slm\n2. slmVectorRAG\n3. slmGraphRAG\n1 or 2 or 3> "
            # )
            model = "3"
            if model.lower() in ("exit", "quit"):
                break

            if model == "slm" or model == "1":
                text = input("> ")

                resp = slm(text)
                message = resp.json()["choices"][0]["message"]["content"]
                print("=" * 60)
                print(message)
            elif model == "slmVectorRAG" or model == "2":
                text = input("> ")

                nodes = graph.search_nodes(text, top_k=5)[:3]
                context = create_context(nodes)
                print(f"context: {context}")
                resp = slm_RAG(text, context=context)
                message = resp.json()["choices"][0]["message"]["content"]
                print("=" * 60)
                print(message)

            elif model == "slmGraphRAG" or model == "3":
                # text = input("> ")
                text = "расскажи про sql-инъекции"
                nodes = graph.search_nodes(text, top_k=5, threshold=0.7)
                print("=" * 60)
                print(nodes)

                relations = graph.expand_nodes(
                    text=text, nodes=nodes, max_depth=1, top_k=10, threshold=1.15
                )
                context = create_graph_context(nodes, relations)
                print("=" * 60)
                print(f"context: {context}")
                resp = slm_GraphRAG(text, context=context)
                message = resp.json()["choices"][0]["message"]["content"]
                print("=" * 60)
                print(message)
                exit(130)
            else:
                print("method does not exist")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt")


if __name__ == "__main__":
    main()
