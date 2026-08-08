# metrics
import src.storage as load
from src.Graph import Graph
from src.LanguageModels import (
    create_context,
    slm,
    slm_RAG,
    create_graph_context,
    gemini_as_judge,
    slm_GraphRAG,
)
from dataclasses import dataclass
import matplotlib.pyplot as plt

from pathlib import Path
import hashlib
import json
from collections import defaultdict

import numpy as np


QUESTIONS = [
    "Что такое SQL-инъекция?",
    "Какие меры защиты позволяют предотвратить SQL-инъекции?",
    "Что такое межсайтовый скриптинг (XSS)?",
    "Как защититься от XSS-атак?",
    "Что такое межсайтовая подделка запросов (CSRF)?",
    "Какие меры предотвращают CSRF-атаки?",
    "Что такое переполнение буфера?",
    "Какие угрозы возникают из-за переполнения буфера?",
    "Что такое удаленное выполнение кода (RCE)?",
    "Какие меры позволяют предотвратить RCE?",
    "Что такое повышение привилегий?",
    "Какие способы предотвращают повышение привилегий?",
    "Что такое отказ в обслуживании (DoS)?",
    "Как защититься от DoS-атак?",
    "Что такое распределенная атака отказа в обслуживании (DDoS)?",
    "Какие существуют методы защиты от DDoS?",
    "Что такое фишинг?",
    "Какие меры помогают защититься от фишинга?",
    "Что такое вредоносное программное обеспечение?",
    "Какие существуют способы защиты от вредоносного ПО?",
    "Что такое контроль доступа?",
    "Какие модели управления доступом существуют?",
    "Что такое принцип минимальных привилегий?",
    "Почему важно использовать принцип минимальных привилегий?",
    "Что такое многофакторная аутентификация?",
    "Какие преимущества дает многофакторная аутентификация?",
    "Что такое журналирование событий безопасности?",
    "Зачем необходимо журналирование событий безопасности?",
    "Что такое мониторинг информационной безопасности?",
    "Какие задачи решает мониторинг безопасности?",
    "Что такое система обнаружения вторжений (IDS)?",
    "Чем IDS отличается от IPS?",
    "Что такое предотвращение вторжений (IPS)?",
    "Какие задачи выполняет IPS?",
    "Что такое сегментация сети?",
    "Зачем необходима сегментация сети?",
    "Что такое криптографическая защита информации?",
    "Для чего используется шифрование данных?",
    "Что такое цифровая подпись?",
    "Какие свойства обеспечивает электронная подпись?",
    "Что такое управление уязвимостями?",
    "Какие этапы включает процесс управления уязвимостями?",
    "Что такое оценка рисков информационной безопасности?",
    "Для чего проводится оценка рисков?",
    "Что такое инцидент информационной безопасности?",
    "Какие этапы включает обработка инцидента?",
    "Что такое резервное копирование?",
    "Почему резервное копирование является мерой защиты информации?",
    "Что такое политика информационной безопасности?",
    "Какие основные цели политики информационной безопасности?",
]
BDU_QUESTIONS = [
    "Что представляет собой блокировка доступа к сайтам или типам сайтов, запрещенных к использованию?",
    "Для чего применяется блокировка доступа к запрещенным сайтам?",
    "Что включает защита беспроводных соединений?",
    "Какие угрозы позволяет снизить защита беспроводных соединений?",
    "Что понимается под информированием о компьютерных инцидентах?",
    "Какие действия включает информирование о компьютерных инцидентах?",
    "Что означает обеспечение возможности восстановления информации?",
    "Почему обеспечение возможности восстановления информации является важной мерой защиты?",
    "Что означает установка только разрешенного к использованию программного обеспечения?",
    "Какие угрозы предотвращает использование только разрешенного программного обеспечения?",
    "Что включает реагирование на обнаружение зараженных объектов информационной системы?",
    "Какие действия выполняются при обнаружении вредоносного программного обеспечения?",
    "Что включает защита информации о событиях безопасности?",
    "Почему необходимо защищать журналы регистрации событий безопасности?",
    "Для чего используется загрузка операционной системы только с носителей, доступных только для чтения?",
    "Какие угрозы снижает загрузка операционной системы с неизменяемых носителей?",
    "Что представляет собой регистрация событий, связанных с получением информации от другого пользователя?",
    "Для чего необходима регистрация событий обмена информацией между пользователями?",
    "Что представляет собой ведение журнала учета машинных носителей информации?",
    "Зачем ведется журнал учета машинных носителей информации?",
    "Когда выполняется модернизация или замена компонентов информационных систем?",
    "Какие задачи решает модернизация компонентов информационной системы?",
    "Что такое шифрование данных?",
    "Какие задачи решает шифрование данных?",
    "Что представляет собой внедрение вредоносного программного обеспечения с помощью файлового архива?",
    "Какие меры позволяют снизить риск заражения через файловые архивы?",
    "Что представляют собой атаки на уровне каналов и сети, приводящие к изменению маршрутов?",
    "Какие последствия могут вызвать атаки, изменяющие маршруты передачи данных?",
    "Что представляет собой атака типа «человек посередине» с использованием поддельной точки доступа?",
    "Какие меры защиты позволяют противодействовать атаке через поддельную точку доступа?",
    "Что представляют собой атаки через социальные сети?",
    "Какие угрозы связаны с использованием социальных сетей?",
    "Что означает выход за пределы замкнутой программной среды?",
    "Какие последствия может вызвать выход за пределы замкнутой программной среды?",
    "Что представляет собой сервер?",
    "Какие функции выполняет сервер в информационной системе?",
    "Что понимается под нарушением личной и семейной тайны?",
    "Какие последствия может иметь нарушение личной и семейной тайны?",
    "Что понимается под публикацией недостоверной информации на веб-ресурсах организации?",
    "Какие последствия может вызвать публикация недостоверной информации на веб-ресурсах организации?",
    "Какую роль играют поставщики вычислительных услуг и услуг связи?",
    "Какие риски могут быть связаны с поставщиками вычислительных услуг и услуг связи?",
    "Какие меры защиты связаны с обеспечением возможности восстановления информации?",
    "Какие меры защиты применяются для обеспечения безопасности журналов регистрации?",
    "Какие угрозы могут быть снижены с помощью шифрования данных?",
    "Какие меры позволяют повысить безопасность беспроводных соединений?",
]
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
    for result in results:
        print("\n")
        print(result)


def plot_table(results: list[ExperimentResult]):
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
                round(r.correctness, 2),
                round(r.completeness, 2),
                round(r.faithfulness, 2),
                round(r.clarity, 2),
                round(r.total, 2),
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


def plot_quality(results: list[ExperimentResult]):
    names = [r.name for r in results]

    metrics = {
        "Correctness": [r.correctness for r in results],
        "Completeness": [r.completeness for r in results],
        "Faithfulness": [r.faithfulness for r in results],
        "Clarity": [r.clarity for r in results],
        "Total": [r.total for r in results],
    }

    x = np.arange(len(names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (label, values) in enumerate(metrics.items()):
        bars = ax.bar(
            x + (i - 2) * width,
            values,
            width,
            label=label,
        )

        ax.bar_label(
            bars,
            fmt="%.2f",
            padding=3,
            fontsize=8,
        )

    ax.set_title("Качество ответов")
    ax.set_ylabel("Оценка")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 10.5)

    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_performance(results: list[ExperimentResult]):
    names = [r.name for r in results]

    time_seconds = [r.total_ms / 1000 for r in results]

    total_tokens = [r.total_tokens for r in results]

    input_tokens = [r.prompt_tokens for r in results]

    output_tokens = [r.completion_tokens for r in results]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    # -------------------------
    # Время ответа
    # -------------------------

    bars = axes[0].bar(
        names,
        time_seconds,
    )

    axes[0].bar_label(
        bars,
        fmt="%.2f s",
        padding=3,
    )

    axes[0].set_title("Среднее время ответа")
    axes[0].set_ylabel("Секунды")
    axes[0].grid(axis="y", alpha=0.3)

    # -------------------------
    # Токены
    # -------------------------

    x = np.arange(len(names))
    width = 0.25

    bars1 = axes[1].bar(
        x - width,
        input_tokens,
        width,
        label="Input",
    )

    bars2 = axes[1].bar(
        x,
        output_tokens,
        width,
        label="Output",
    )

    bars3 = axes[1].bar(
        x + width,
        total_tokens,
        width,
        label="Total",
    )

    axes[1].bar_label(
        bars1,
        padding=3,
        fontsize=8,
    )

    axes[1].bar_label(
        bars2,
        padding=3,
        fontsize=8,
    )

    axes[1].bar_label(
        bars3,
        padding=3,
        fontsize=8,
    )

    axes[1].set_title("Использование токенов")
    axes[1].set_ylabel("Количество токенов")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

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


def evaluate_question(graph: Graph, text: str):
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
        judge = gemini_as_judge(
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

        print(f"\nVector context:\n{context}")
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
        judge = gemini_as_judge(
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
        nodes = graph.search_nodes(text, top_k=2)

        expanded = graph.expand_nodes(text=text, nodes=nodes, top_k=3)

        context = create_graph_context(
            nodes,
            expanded,
        )

        print(f"\nGraph context:\n{context}")
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
        judge = gemini_as_judge(
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

    return table


def average_results(results: list[ExperimentResult]) -> ExperimentResult:
    n = len(results)

    return ExperimentResult(
        name=results[0].name,
        correctness=sum(r.correctness for r in results) / n,
        completeness=sum(r.completeness for r in results) / n,
        faithfulness=sum(r.faithfulness for r in results) / n,
        clarity=sum(r.clarity for r in results) / n,
        total=sum(r.total for r in results) / n,
        comment="",
        prompt_tokens=sum(r.prompt_tokens for r in results) // n,
        completion_tokens=sum(r.completion_tokens for r in results) // n,
        total_tokens=sum(r.total_tokens for r in results) // n,
        prompt_ms=sum(r.prompt_ms for r in results) / n,
        predicted_ms=sum(r.predicted_ms for r in results) / n,
        total_ms=sum(r.total_ms for r in results) / n,
    )


def main():
    config = load.config()
    files = config["files"]

    graph = Graph(
        files["processed"]["nodes"],
        files["processed"]["edges"],
        files["index"]["nodes"],
        files["embeddings"]["nodes"],
    )

    # text = "Какие меры защиты позволяют предотвратить угрозы, возникающие из-за SQL-инъекций?"

    metrics = defaultdict(list)

    for i, question in enumerate(BDU_QUESTIONS, start=1):
        print(f"[{i}/{len(BDU_QUESTIONS)}] {question}")
        print(question)

        results = evaluate_question(graph, question)

        for r in results:
            metrics[r.name].append(r)

    final_results = []

    for method in ["SLM", "VectorRAG", "GraphRAG"]:
        final_results.append(average_results(metrics[method]))

    print_table(final_results)

    # table = evaluate_question(graph, text)
    # print_comments(table)
    # print_table(final_results)
    plot_table(final_results)
    plot_quality(final_results)
    plot_performance(final_results)


if __name__ == "__main__":
    main()
