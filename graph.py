# build knowlidge base index error with mind sell orders does not have asset idand search in graph

import json
from collections import defaultdict, deque
import scripts as scripts


class GraphSearch:
    def __init__(self, nodes_file: str, edges_file: str):
        self.nodes = self._load(nodes_file)
        self.edges = self._load(edges_file)
        self.graph = defaultdict(list)
        config = scripts.load_config()
        self.relations = config.get("en-ru")
        self._index_edges()

    def _load(self, path: str):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _index_edges(self):
        print("indexing edges ...")
        for edge in self.edges:
            self.graph[edge["source"]].append(edge)
            self.graph[edge["target"]].append(edge)
        print("complite")

    def input(self, text: str):
        words = self._normalize(text)

        print(f"[DEBUG] input: {text}")
        print(f"[DEBUG] words: {words}")

        return words

    def _normalize(self, text: str):
        return [word.lower() for word in text.split() if word.strip()]

    def search(self, tokens: list[str]):
        results = []

        for node in self.nodes:
            score = 0

            node_id = str(node.get("id", "")).lower()
            node_type = str(node.get("type", "")).lower()
            node_name = str(node.get("name", "")).lower()

            for token in tokens:
                if token in node_id:
                    score += 3

                if token in node_type:
                    score += 2

                if token in node_name:
                    score += 5

            if score > 0:
                results.append(
                    {
                        "score": score,
                        "node": node,
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)

        return results

    def bfs(self, start_id: str, max_depth: int = 3):
        """
        Возвращает все ребра в радиусе max_depth хопов от start_id.
        """
        print("start have", len(self.graph[start_id]), "edges")

        visited_nodes = {start_id}
        visited_edges = set()

        result = []

        queue = deque([(start_id, 0)])

        while queue:
            current_node, depth = queue.popleft()

            if depth >= max_depth:
                continue

            for edge in self.graph.get(current_node, []):
                edge_key = (
                    edge["source"],
                    edge["relation"],
                    edge["target"],
                )

                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    result.append(edge)

                # определяем соседа
                if edge["source"] == current_node:
                    neighbor = edge["target"]
                else:
                    neighbor = edge["source"]

                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return result

    def get(self, edges: list[dict]):
        """
        По списку ребер возвращает:
        - все уникальные вершины
        - текстовое представление графа
        """

        # быстрый поиск вершины по id
        node_index = {node["id"]: node for node in self.nodes}

        used_nodes = {}
        lines = []

        for edge in edges:
            source = node_index.get(edge["source"])
            target = node_index.get(edge["target"])

            if source:
                used_nodes[source["id"]] = source

            if target:
                used_nodes[target["id"]] = target

            source_name = source["name"] if source else edge["source"]

            target_name = target["name"] if target else edge["target"]

            edge_type = edge["relation"].lower()
            edge_relation = self.relations.get(edge_type)

            lines.append(
                f"{source_name} ({edge['source']}) "
                f"--{edge_relation}--> "
                f"{target_name} ({edge['target']})"
            )

        return {
            "nodes": list(used_nodes.values()),
            "text": "\n".join(lines),
        }


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
            nodes = graph.search(tokens)
            print(f"\nlen nodes: {len(nodes)}\n")

            all_edges = {}

            for item in nodes:
                node = item["node"]
                id = node["id"]
                print(f"node: {id}")
                edges = graph.bfs(node["id"], max_depth=1)
                context = graph.get(edges)
                # print(context["text"])
                all_edges[id] = edges
                print(f"len edges: {len(edges)}")
            # print(f"\nedges: {all_edges}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt")


if __name__ == "__main__":
    main()
