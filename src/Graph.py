from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

import src.models as obj
from src.LanguageModels import calc_embedding


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class Graph:
    def __init__(
        self,
        nodes_file: str | Path,
        edges_file: str | Path,
        index_file: str | Path,
        embeddings_file: str | Path,
    ):

        self.nodes = self._load_nodes(nodes_file)
        self.edges = self._load_edges(edges_file)

        self.nodes_by_id: dict[int, obj.Node] = {node.id: node for node in self.nodes}

        self.adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)

        self._build_adjacency()

        self.index_file = Path(index_file)
        self.embeddings_file = Path(embeddings_file)

        self.faiss_index: Optional[faiss.Index] = None

        self.embeddings: list[dict] = []

        self._load_vector_index()

    # =====================================================
    # PUBLIC API
    # =====================================================

    def search_nodes(
        self,
        text: str,
        top_k: int = 5,
        threshold: float | None = 0.7,
    ) -> list[obj.Node]:

        if self.faiss_index is None:
            raise RuntimeError("FAISS index is not loaded")

        vector = np.array([calc_embedding(text)], dtype="float32")

        distances, indexes = self.faiss_index.search(vector, self.faiss_index.ntotal)

        result = []

        for idx, distance in zip(indexes[0], distances[0]):
            if idx < 0:
                continue

            if threshold is not None and distance > threshold:
                continue

            node = self.faiss_id_to_node[self.embeddings[idx]["id"]]
            result.append(node)

            if len(result) == top_k:
                break

        return result

    def expand_nodes(
        self,
        text: str,
        nodes: list[obj.Node],
        max_depth: int = 1,
        top_k: int = 5,
        threshold: float = 1.1,
    ) -> dict[int, list[dict]]:

        query_embedding = calc_embedding(text)
        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        print("\nStart nodes:")
        for node in nodes:
            print(f"  {node.id} | {node.internal_id} | {node.name}")

        # ---------- собираем кандидатов ----------

        candidates: dict[int, tuple[obj.Node, str, int]] = {}

        queue = deque((node.id, 0) for node in nodes)
        visited = {node.id for node in nodes}

        while queue:
            node_id, depth = queue.popleft()

            root = self.nodes_by_id[node_id]

            if depth >= max_depth:
                print("  depth limit reached")
                continue

            for neighbour_id, relation in self.adjacency[node_id]:
                neighbour = self.nodes_by_id.get(neighbour_id)

                if neighbour is None:
                    continue

                if neighbour.id not in candidates:
                    candidates[neighbour.id] = (
                        neighbour,
                        relation,
                        node_id,
                    )

                if neighbour.id not in visited:
                    visited.add(neighbour.id)
                    queue.append((neighbour.id, depth + 1))

        print("\nCandidates:", len(candidates))

        # ---------- similarity ----------

        scored = []

        print("\nSimilarity:")

        for node_id, (node, relation, root_id) in candidates.items():
            embedding = self.embedding_by_id[node_id]

            distance = np.linalg.norm(query_embedding - embedding)

            print(f"{distance:.4f} | {node.internal_id} | {node.name}")

            if distance <= threshold:
                print("   ACCEPT")

                scored.append(
                    (
                        distance,
                        root_id,
                        relation,
                        node,
                    )
                )
            else:
                print("   REJECT")

        scored.sort(key=lambda x: x[0])

        print("\nAccepted after sorting:")

        for distance, root_id, relation, node in scored:
            root = self.nodes_by_id[root_id]

            print(
                f"{distance:.4f} | "
                f"{root.internal_id} -> "
                f"{relation} -> "
                f"{node.internal_id}"
            )

        result = defaultdict(list)

        print("\nFinal result:")

        for distance, root_id, relation, node in scored[:top_k]:
            root = self.nodes_by_id[root_id]

            print(f"{distance:.4f} | {root.name} -> {relation} -> {node.name}")

            result[root_id].append(
                {
                    "node": node,
                    "relation": relation,
                }
            )

        print("=" * 80)

        return dict(result)

    # =====================================================
    # PRIVATE
    # =====================================================

    def _build_adjacency(self):

        for edge in self.edges:
            self.adjacency[edge.source].append(
                (
                    edge.target,
                    edge.relation,
                )
            )

            self.adjacency[edge.target].append(
                (
                    edge.source,
                    edge.relation,
                )
            )

        logger.info(
            "Graph loaded: nodes=%d edges=%d",
            len(self.nodes_by_id),
            len(self.edges),
        )

    def _load_vector_index(self):

        if not self.index_file.exists():
            raise FileNotFoundError(self.index_file)

        with open(self.embeddings_file, encoding="utf-8") as f:
            self.embeddings = json.load(f)

        if not self.embeddings_file.exists():
            raise FileNotFoundError(self.embeddings_file)
        self.embedding_by_id = {}
        self.faiss_id_to_node = {}

        self.embedding_by_id = {
            item["id"]: np.asarray(item["embedding"], dtype=np.float32)
            for item in self.embeddings
        }

        self.faiss_id_to_node = {
            item["id"]: self.nodes_by_id[item["id"]] for item in self.embeddings
        }
        self.faiss_index = faiss.read_index(str(self.index_file))

        logger.info("FAISS loaded: %d vectors", self.faiss_index.ntotal)

    def _load_nodes(
        self,
        path: str | Path,
    ) -> list[obj.Node]:

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return [obj.Node(**item) for item in data]

    def _load_edges(
        self,
        path: str | Path,
    ) -> list[obj.Edge]:

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return [obj.Edge(**item) for item in data]
