# slm, llm and other connections with LanguageModel
import requests
import src.models as obj
import json


SYSTEM = """
Ты являешься экспертом по информационной безопасности.

Тебе предоставлен контекст.

Используй его как основной источник информации.

Правила:

1. Все утверждения должны подтверждаться контекстом.
2. Не придумывай новые объекты и связи.
3. Если контекст не содержит ответа, прямо сообщи об этом.
4. Не добавляй собственные знания, если контекст уже позволяет ответить.
5. Если используешь собственные знания из-за отсутствия информации в контексте, явно отметь это.

Ответ должен быть кратким и информативным.
"""


def create_context(nodes: list[obj.Node]):
    contexts = []
    for item in nodes:
        string = f"({item.internal_id}) {item.name}"
        if item.description != "":
            string = string + f":\n {item.description}"
        contexts.append(string)

    return "\n".join(contexts)


def create_graph_context(nodes: list[obj.Node], expanded: dict[int, list[dict]]) -> str:

    contexts = []

    for node_id, relations in expanded.items():
        lines = []
        base_context = None
        for item in nodes:
            if item.id == node_id:
                base_context = f"({item.internal_id}) {item.name}"
                if item.description != "":
                    base_context = base_context + f"\n{item.description}"
                base_context = base_context + ":"
        if base_context is None:
            print(f"nodes: {nodes}")
            print(f"node_id: {node_id}, type: {type(node_id)}")
            raise ReferenceError("node is not found")

        for item in relations:
            node = item["node"]
            relation = item["relation"]

            lines.append(f"- {relation}: ({node.internal_id}) {node.name}")

        contexts.append(f"{base_context}:\n" + "\n".join(lines))

    return "\n\n".join(contexts)


def calc_embedding(sentence):
    response = requests.post(
        "http://localhost:8081/embedding",
        json={"content": sentence},
    )
    return response.json()[0]["embedding"][0]


def slm_RAG(text, context):

    prompt = f"""
Контекст:

{context}

Вопрос:

{text}
"""

    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen",
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        },
    )

    return response


def slm_GraphRAG(text, context):

    prompt = f"""
Контекст:

{context}

Вопрос:

{text}
"""

    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen",
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        },
    )

    return response


def slm(text):

    system = """
Ты являешься экспертом по информационной безопасности.

Отвечай максимально точно и структурированно.

Если не уверен в факте — сообщи об этом.

Не выдумывай информацию.
"""
    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen",
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        },
    )

    return response


def llm_as_judge(
    question: str,
    answer: str,
    context: str,
):
    system = """
Ты — независимый эксперт по информационной безопасности.

Оцени ответ по критериям:

correctness — фактическая корректность.
completeness — полнота ответа.
faithfulness — соответствие предоставленному контексту.
clarity — понятность и структура.

Faithfulness:
10 — все утверждения подтверждены контекстом.
7–9 — почти все.
4–6 — часть не подтверждается.
1–3 — большая часть отсутствует в контексте.
Если контекст пустой — faithfulness = 0.

Не исправляй ответ.
коментарий не более 25 слов

Верни только JSON:

{
  "correctness": 0,
  "completeness": 0,
  "faithfulness": 0,
  "clarity": 0,
  "comment": ""
}

"""
    user = {
        "question": question,
        "context": context,
        "answer": answer,
    }
    # print(json.dumps(user, ensure_ascii=False, indent=2))

    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen",
            "temperature": 0,
            "top_p": 0.1,
            "max_tokens": 128,
            "response_format": {
                "type": "json_object",
            },
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        user,
                        ensure_ascii=False,
                    ),
                },
            ],
        },
    )

    try:
        response.raise_for_status()
        result = json.loads(response.json()["choices"][0]["message"]["content"])

        result["total"] = round(
            (result["correctness"] + result["completeness"] + result["clarity"]) / 3,
            1,
        )

        return result
    except Exception as e:
        print(f"exception: {e}")
        print("data:")
        print(user)
        print("response:")
        print(response.status_code)
        print(response.text)
        exit(130)
