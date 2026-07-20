import src.models as obj
from src.LanguageModels import slm_crag, slm_graph_crag


class Crag:
    def validate_nodes(
        self, query: str, nodes: list[obj.Node], threshold: float
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
        max_relations: int = 5,
    ) -> dict[int, list[dict]]:
        """
        Удаляет бесполезные связи.
        """

        contexts = []

        for node_id, relation in relations.items():
            root = None

            for item in root_nodes:
                if item.id == node_id:
                    root = item
                    break

            if root is None:
                raise ReferenceError(f"Node {node_id} not found")

            context = {
                "root_id": root.internal_id,
                "root_name": root.name,
                "relations": [],
            }

            for index, item in enumerate(relation):
                node = item["node"]

                context["relations"].append(
                    {
                        "index": index,
                        "relation": item["relation"],
                        "node_id": node.internal_id,
                        "node_name": node.name,
                    }
                )

            contexts.append(context)

        print(contexts)
        result = slm_graph_crag(
            contexts=contexts,
            query=query,
        )
        print("=" * 60)
        print(result)
        exit(130)

    def convert_int_ids(self, nodes, arr_ids):
        result = []
        for internal_id in arr_ids:
            for node in nodes:
                if node.internal_id == internal_id:
                    result.append(node)
        return result

