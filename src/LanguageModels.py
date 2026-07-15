# slm, llm and other connections with LanguageModel
import requests
import src.models as obj
import json


def create_context(nodes: list[obj.Node]):
    contexts = []
    for item in nodes:
        string = f"({item.id}) {item.name}"
        contexts.append(string)

    return "\n".join(contexts)


def create_graph_context(expanded: dict[str, list[dict]]) -> str:

    contexts = []

    for node_id, relations in expanded.items():
        lines = []

        for item in relations:
            node = item["node"]
            relation = item["relation"]

            lines.append(f"- {relation}: ({node.id}) {node.name}")

        contexts.append(f"{node_id}:\n" + "\n".join(lines))

    return "\n\n".join(contexts)


def calc_embedding(sentence):
    response = requests.post(
        "http://localhost:8081/embedding",
        json={"content": sentence},
    )
    return response.json()[0]["embedding"][0]


def slm_RAG(text, context):
    prompt = f"""
Ты ассистент подключённый к базе знаний БДУ ФСТЕК(https://bdu.fstec.ru).
Используй только предоставленный контекст когда контекст валиден к ответу.

Контекст:
{context}

Вопрос:
{text}

Ответ:
"""

    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": prompt}],
        },
    )

    return response


def slm(text):
    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": text}],
        },
    )

    return response


def llm_as_judge(
    questions: list[str],
    answers: list[str],
    contexts: list[str],
):
    if not (len(questions) == len(answers) == len(contexts)):
        raise ValueError("questions, answers and contexts must have the same length")

    content = ""

    for i, (question, answer, context) in enumerate(
        zip(questions, answers, contexts),
        start=1,
    ):
        content += f"""
=== Ответ {i} ===

Вопрос:
{question}

Контекст:
{context}

Ответ модели:
{answer}

"""

    prompt = f"""
Ты выступаешь в роли независимого эксперта по информационной безопасности.

Твоя задача независимо оценить каждый ответ.

Для каждого ответа оцени по шкале от 1 до 10:

- correctness — фактическая корректность ответа;
- completeness — полнота ответа;
- faithfulness — насколько ответ соответствует предоставленному контексту и не придумывает информацию;
- clarity — понятность и качество изложения.

После оценки вычисли итоговую оценку total.

Верни ТОЛЬКО JSON следующего вида:

[
    {{
    "id": 1,
    "correctness": 0,
    "completeness": 0,
    "faithfulness": 0,
    "clarity": 0,
    "total": 0,
    "comment": ""
    }},
]

Данные для оценки:

{content}
"""

    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0,
        },
    )

    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    print(content)
    return json.loads(content)
