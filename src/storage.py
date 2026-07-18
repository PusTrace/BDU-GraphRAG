from logging import getLogger
import json
import csv
from pathlib import Path

import src.models as obj

log = getLogger("storage")
CONFIG_FILE = "config.json"


def nodes(path: str) -> list[obj.Node]:
    raw_items = _json(path)
    nodes = []
    for item in raw_items:
        try:
            nodes.append(obj.Node.from_dict(item))
        except ValueError as exc:
            log.warning("пропускаю некорректный узел: %s", exc)
    return nodes


def edges(path: str | Path) -> list[obj.Edge]:
    raw_items = _json(path)
    edges = []
    for item in raw_items:
        try:
            edges.append(obj.Edge.from_dict(item))
        except ValueError as exc:
            log.warning("пропускаю некорректное ребро: %s", exc)
    return edges


def _json(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"файл не найден: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def config(path: str | Path = CONFIG_FILE) -> dict:
    config = _json(path)
    return config


def result(path: str | Path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)
