from dataclasses import dataclass


@dataclass
class Node:
    id: str
    name: str
    type: str = ""


@dataclass
class Edge:
    source: str
    target: str
    relation: str


@dataclass
class ScoredNode:
    score: float
    node: Node
