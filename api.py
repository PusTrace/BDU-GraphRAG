# api for chat with llm
import json

from Graph import GraphSearch
from llm import llm, llm_without_graph


def terminal_llm():
    graph = GraphSearch()
    graph.init(
        "data/nodes.json",
        "data/edges.json",
    )
    try:
        while True:
            text = input("> ")

            if text.lower() in ("exit", "quit"):
                break

            context = graph.lazy(text)

            resp = llm(user_input=text, context=context)
            message = resp.json()["choices"][0]["message"]["content"]
            usage = resp.json()["usage"]
            timings = resp.json()["timings"]
            print(message)
            print(json.dumps(usage, ensure_ascii=False, indent=4))
            print(json.dumps(timings, ensure_ascii=False, indent=4))

    except KeyboardInterrupt:
        print("KeyboardInterrupt")


def terminal_llm_without_graph():
    try:
        while True:
            text = input("> ")

            resp = llm_without_graph(user_input=text)
            message = resp.json()["choices"][0]["message"]["content"]
            usage = resp.json()["usage"]
            timings = resp.json()["timings"]
            print(message)
            print(json.dumps(usage, ensure_ascii=False, indent=4))
            print(json.dumps(timings, ensure_ascii=False, indent=4))
    except KeyboardInterrupt:
        print("KeyboardInterrupt")


if __name__ == "__main__":
    # terminal_llm_without_graph()
    terminal_llm()
