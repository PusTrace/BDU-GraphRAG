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

        self.nodes_by_id: dict[str, obj.Node] = {node.id: node for node in self.nodes}

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
        threshold: float | None = None,
    ) -> list[obj.Node]:

        if self.faiss_index is None:
            raise RuntimeError("FAISS index is not loaded")

        vector = np.array([calc_embedding(text)], dtype="float32")

        distances, indexes = self.faiss_index.search(vector, top_k)

        result = []

        for idx, distance in zip(indexes[0], distances[0]):
            if idx < 0:
                continue

            item = self.embeddings[idx]

            if threshold is not None:
                if distance > threshold:
                    continue

            node_id = item["id"]

            node = self.nodes_by_id.get(node_id)

            if node:
                result.append(node)

        return result

    def expand_nodes(
        self,
        nodes: list[obj.Node],
        max_depth: int = 1,
        max_neighbors: int | None = None,
    ) -> dict[str, list[dict]]:

        result = defaultdict(list)

        queue = deque([(node.id, 0) for node in nodes])

        visited = {node.id for node in nodes}

        while queue:
            node_id, depth = queue.popleft()

            if depth >= max_depth:
                continue

            neighbors_count = 0

            for neighbour_id, relation in self.adjacency.get(node_id, []):
                if max_neighbors is not None:
                    if neighbors_count >= max_neighbors:
                        break

                neighbour = self.nodes_by_id.get(neighbour_id)

                if neighbour is None:
                    continue

                result[node_id].append(
                    {
                        "node": neighbour,
                        "relation": relation,
                    }
                )

                neighbors_count += 1

                if neighbour_id not in visited:
                    visited.add(neighbour_id)

                    queue.append(
                        (
                            neighbour_id,
                            depth + 1,
                        )
                    )

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

        if not self.embeddings_file.exists():
            raise FileNotFoundError(self.embeddings_file)

        self.faiss_index = faiss.read_index(str(self.index_file))

        with open(self.embeddings_file, encoding="utf-8") as f:
            self.embeddings = json.load(f)

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
