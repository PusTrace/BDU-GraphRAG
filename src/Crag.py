import src.models as obj
from src.LanguageModels import slm_crag, slm_graph_crag


class Crag:
    def validate_nodes(
        self, query: str, nodes: list[obj.Node], threshold: float = 0.7
    ) -> list[obj.Node]:
        """
        Удаляет нерелевантные вершины.
        """
        context = ""
        count = 0
        for node in nodes:
            count += 1
            context += f"{count}.\nID:{node.internal_id}\nName:{node.name}\nDescription:{node.description}\n\n"

        result = slm_crag(query, context)
        print(result)
        scores = result.get("scores")
        while True:
            node_ids = []
            for item in scores:
                if float(item["score"]) > threshold:
                    node_ids.append(item["id"])
            if len(node_ids) > 0:
                return self.convert_int_ids(nodes, node_ids)
            else:
                threshold += 0.15

    def validate_relations(
        self,
        query: str,
        root_nodes: list[obj.Node],
        relations: dict[int, list[dict]],
        threshold: float = 0.7,
        max_relations: int = 10,
    ) -> dict[int, list[dict]]:

        contexts = []

        for node_id, relation in relations.items():
            root = next((n for n in root_nodes if n.id == node_id), None)

            if root is None:
                raise ReferenceError(f"Node {node_id} not found")

            contexts.append(
                {
                    "root_id": root.internal_id,
                    "root_name": root.name,
                    "relations": [
                        {
                            "index": i,
                            "relation": r["relation"],
                            "node_id": r["node"].internal_id,
                            "node_name": r["node"].name,
                        }
                        for i, r in enumerate(relation)
                    ],
                }
            )

        result = slm_graph_crag(
            contexts=contexts,
            query=query,
        )

        filtered: dict[int, list[dict]] = {}

        for root_result in result["scores"]:
            root_internal_id = root_result["root_id"]

            # ищем id вершины по internal_id
            root = next(
                (n for n in root_nodes if n.internal_id == root_internal_id),
                None,
            )

            if root is None:
                continue

            original = relations[root.id]

            keep = []

            for relation_score in root_result["relations"]:
                if relation_score["score"] < threshold:
                    continue

                idx = relation_score["index"]

                if idx >= len(original):
                    continue

                keep.append(original[idx])

            if max_relations is not None:
                keep = keep[:max_relations]

            filtered[root.id] = keep

        return filtered

    def convert_int_ids(self, nodes, arr_ids):
        result = []
        for internal_id in arr_ids:
            for node in nodes:
                if node.internal_id == internal_id:
                    result.append(node)
        return result
