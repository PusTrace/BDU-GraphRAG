# metrics
import src.storage as load
from src.Graph import Graph
from src.LanguageModels import (
    create_context,
    slm,
    slm_RAG,
    create_graph_context,
    llm_as_judge,
    slm_GraphRAG,
)
from dataclasses import dataclass
import matplotlib.pyplot as plt

from pathlib import Path
import hashlib
import json


CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(exist_ok=True)


def cache_path(text: str, stage: str) -> Path:
    key = hashlib.sha256(f"{text}:{stage}".encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.json"


def load_cache(text: str, stage: str):
    path = cache_path(text, stage)

    if not path.exists():
        return None

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_cache(text: str, stage: str, data):
    path = cache_path(text, stage)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


@dataclass
class ExperimentResult:
    name: str

    correctness: float
    completeness: float
    faithfulness: float
    clarity: float
    total: float
    comment: str

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    prompt_ms: float
    predicted_ms: float
    total_ms: float


def print_table(results: list[ExperimentResult]):
    headers = [
        "Метод",
        "Correct",
        "Complete",
        "Faith",
        "Clear",
        "Total",
        "inp tok",
        "out tok",
        "Total tok",
        "Time (ms)",
    ]

    rows = []

    for r in results:
        rows.append(
            [
                r.name,
                r.correctness,
                r.completeness,
                r.faithfulness,
                r.clarity,
                r.total,
                r.prompt_tokens,
                r.completion_tokens,
                r.total_tokens,
                round(r.total_ms, 1),
            ]
        )

    fig, ax = plt.subplots(figsize=(13, 2 + len(rows) * 0.6))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.7)

    plt.tight_layout()
    plt.show()


def print_comments(results: list[ExperimentResult]):
    print("\n" + "=" * 80)
    print("Комментарии LLM-судьи")
    print("=" * 80)

    for r in results:
        print(f"\n[{r.name}]")
        print("-" * 80)
        print(r.comment)


def build_result(llm_result, usages, timings):
    names = [
        "SLM",
        "VectorRAG",
        "GraphRAG",
    ]

    result = []

    for name, judge, usage, timing in zip(
        names,
        llm_result,
        usages,
        timings,
    ):
        result.append(
            ExperimentResult(
                name=name,
                correctness=judge["correctness"],
                completeness=judge["completeness"],
                faithfulness=judge["faithfulness"],
                clarity=judge["clarity"],
                total=judge["total"],
                comment=judge["comment"],
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
                prompt_ms=round(timing["prompt_ms"], 2),
                predicted_ms=round(timing["predicted_ms"], 2),
                total_ms=round(
                    timing["prompt_ms"] + timing["predicted_ms"],
                    2,
                ),
            )
        )

    return result


def main():
    config = load.config()
    files = config["files"]

    graph = Graph(
        files["processed"]["nodes"],
        files["processed"]["edges"],
        files["index"]["nodes"],
        files["embeddings"]["nodes"],
    )

    text = "расскажи про sql-инъекции"

    usages = []
    timings = []
    results = []

    # ==========================================================
    # SLM
    # ==========================================================

    cached = load_cache(text, "slm")

    if cached is None:
        resp = slm(text).json()
        save_cache(text, "slm", resp)
    else:
        resp = cached

    answer = resp["choices"][0]["message"]["content"]

    usages.append(resp["usage"])
    timings.append(resp["timings"])

    cached = load_cache(text, "judge_slm")
    if cached is None:
        judge = llm_as_judge(
            question=text,
            answer=answer,
            context="",
        )
        save_cache(text, "judge_slm", judge)
    else:
        judge = cached

    results.append(judge)

    # ==========================================================
    # Vector RAG
    # ==========================================================

    cached = load_cache(text, "vector")

    if cached is None:
        nodes = graph.search_nodes(text)
        context = create_context(nodes)

        resp = slm_RAG(text, context).json()

        save_cache(
            text,
            "vector",
            {
                "context": context,
                "response": resp,
            },
        )
    else:
        context = cached["context"]
        resp = cached["response"]

    answer = resp["choices"][0]["message"]["content"]

    usages.append(resp["usage"])
    timings.append(resp["timings"])

    cached = load_cache(text, "judge_vector")
    if cached is None:
        judge = llm_as_judge(
            question=text,
            answer=answer,
            context=context,
        )
        save_cache(text, "judge_vector", judge)
    else:
        judge = cached

    results.append(judge)

    # ==========================================================
    # Graph RAG
    # ==========================================================

    cached = load_cache(text, "graph")

    if cached is None:
        nodes = graph.search_nodes(text)

        expanded = graph.expand_nodes(
            text=text,
            nodes=nodes,
        )

        context = create_graph_context(
            nodes,
            expanded,
        )

        resp = slm_GraphRAG(text, context).json()

        save_cache(
            text,
            "graph",
            {
                "context": context,
                "response": resp,
            },
        )
    else:
        context = cached["context"]
        resp = cached["response"]

    answer = resp["choices"][0]["message"]["content"]

    usages.append(resp["usage"])
    timings.append(resp["timings"])

    cached = load_cache(text, "judge_graph")
    if cached is None:
        judge = llm_as_judge(
            question=text,
            answer=answer,
            context=context,
        )
        save_cache(text, "judge_graph", judge)
    else:
        judge = cached

    results.append(judge)

    # ==========================================================
    # Report
    # ==========================================================

    table = build_result(
        results,
        usages,
        timings,
    )

    print_comments(table)
    print_table(table)


if __name__ == "__main__":
    main()
