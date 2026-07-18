from dataclasses import dataclass


@dataclass
class Node:
    id: int
    internal_id: str
    name: str
    description: str = ""
    type: str = ""


@dataclass
class Edge:
    source: int
    target: int
    relation: str


@dataclass
class ScoredNode:
    score: float
    node: Node
