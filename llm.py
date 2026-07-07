# llm

import requests
import json

from graph import GraphSearch


def llm(user_input, context):
    prompt = f"""
Ты отвечаешь на вопросы по базе знаний БДУ ФСТЕК(https://bdu.fstec.ru).
Используй только предоставленный контекст.

Контекст:
{context}

Вопрос:
{user_input}

Ответ:
"""

    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": prompt}],
        },
    )

    return response


def llm_without_graph(user_input):
    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": user_input}],
        },
    )

    return response


def main():

    graph = GraphSearch(
        "data/nodes.json",
        "data/edges.json",
    )
    try:
        while True:
            text = input("> ")

            if text.lower() in ("exit", "quit"):
                break

            tokens = graph.input(text)
            nodes = graph.search(tokens)[:3]
            print(f"\nlen nodes: {len(nodes)}\n")

            all_edges = {}
            contexts = []

            for item in nodes:
                node = item["node"]
                id = node["id"]
                print(f"node: {id}")
                edges = graph.bfs(node["id"], max_depth=1)[:10]
                context = graph.get(edges)
                contexts.append(context["text"])
                # print(context["text"])
                all_edges[id] = edges
                print(f"len edges: {len(edges)}")
            # print(f"\nedges: {all_edges}")

            print(contexts)
            full_context = "\n\n".join(contexts)

            resp = llm(user_input=text, context=full_context)
            print("with graph:")
            print(json.dumps(resp.json(), ensure_ascii=False, indent=4))
            print(resp.json()["choices"][0]["message"]["content"])

            resp = llm_without_graph(user_input=text)
            print("\n\nwithout graph:")
            print(json.dumps(resp.json(), ensure_ascii=False, indent=4))
            print(resp.json()["choices"][0]["message"]["content"])
    except KeyboardInterrupt:
        print("KeyboardInterrupt")


if __name__ == "__main__":
    main()
