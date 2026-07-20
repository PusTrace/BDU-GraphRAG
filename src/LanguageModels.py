# slm, llm and other connections with LanguageModel
import requests
import src.models as obj
import json


def create_context(nodes: list[obj.Node]):
    contexts = []
    for item in nodes:
        string = f"({item.internal_id}) {item.name}"
        if item.description != "":
            string = string + f":\n {item.description}"
        contexts.append(string)

    return "\n".join(contexts)


def create_graph_context(nodes: list[obj.Node], expanded: dict[str, list[dict]]) -> str:

    contexts = []

    for node_id, relations in expanded.items():
        lines = []
        base_context = None
        for item in nodes:
            if item.id == node_id:
                base_context = f"({item.internal_id}) {item.name}"
                if item.description != "":
                    base_context = base_context + f":\n {item.description}"
        if base_context is None:
            print(f"nodes: {nodes}, type item.id:{type(item.id)}")
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


def slm_crag(query, nodes):
    system = """
Ты оцениваешь качество поиска GraphRAG.

Тебе будут даны:
- вопрос пользователя;
- список найденных вершин графа.

Для каждой вершины оцени, насколько она релевантна вопросу.

Оценка:
1.0 — вершина напрямую отвечает на вопрос.
0.8 — очень полезна для ответа.
0.5 — частично связана с вопросом.
0.2 — слабо связана.
0.0 — не относится к вопросу.

Используй только информацию из вопроса и текста вершины.

Верни ТОЛЬКО JSON следующего вида:

{
  "scores": [
    {
      "id": "...",
      "score": 0.95,
      "reason": "..."
    }
  ]
}

Не добавляй никакого текста вне JSON.
"""

    prompt = f"""
Вопрос пользователя:

{query}

Найденные вершины:

{nodes}
"""

    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen",
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        },
    )

    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def slm_graph_crag(contexts, query):
    system = """
Ты оцениваешь полезность связей в графе знаний для ответа на вопрос пользователя.

Тебе передаются:
- вопрос пользователя;
- список вершин графа и их связей.

Твоя задача — оценить каждую связь отдельно.

Оценивай только связь между центральной вершиной и соседней вершиной.

Шкала оценки:

1.0 — связь напрямую необходима для ответа.
0.8 — связь сильно помогает раскрыть вопрос.
0.5 — связь частично полезна.
0.2 — связь имеет слабое отношение.
0.0 — связь бесполезна для данного вопроса.

При оценке учитывай:
- смысл вопроса;
- название центральной вершины;
- тип отношения;
- название связанной вершины.

Не используй внешние знания.
Не меняй структуру данных.
Не добавляй новые связи.

Верни ТОЛЬКО JSON.

Формат ответа:

{
  "scores": [
    {
      "root_id": "ID центральной вершины",
      "relations": [
        {
          "index": 0,
          "score": 0.0,
          "reason": "краткое объяснение"
        }
      ]
    }
  ]
}
"""

    payload = {
        "query": query,
        "contexts": contexts,
    }

    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen",
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
        },
    )

    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]

    return json.loads(content)


def slm_RAG(text, context):

    system = """
Ты ассистент по информационной безопасности,
использующий базу знаний БДУ ФСТЭК.

Правила:

1. Контекст из базы знаний является основным источником информации.
2. Не придумывай связи между объектами, которых нет в контексте.
3. Если информации недостаточно, явно скажи об этом.
4. Общие знания по информационной безопасности можно использовать,
но они должны быть отделены от информации из базы знаний.
5. Не утверждай наличие связей, если они не указаны в контексте.

При ответе:
- сначала объясни объект из контекста;
- затем перечисли связанные элементы;
- затем объясни практическое значение.

Формат:

## Определение

## Связанные элементы БДУ

## Объяснение

## Меры защиты

"""

    prompt = f"""
Контекст БДУ:

{context}


Вопрос пользователя:

{text}


Ответ:
"""

    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen",
            "messages": [
                {
                    "role": "system",
                    "content": system,
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

    system = """
Ты являешься аналитиком информационной безопасности.

Тебе предоставлен фрагмент графа знаний БДУ ФСТЭК.

Контекст содержит:
- узлы (объекты);
- идентификаторы;
- связи между объектами.

Правила работы:

1. Используй связи графа как источник истины.
2. Не создавай новые связи между узлами.
3. Если связь отсутствует в графе, не утверждай её наличие.
4. Можно использовать общие знания ИБ для объяснения терминов.
5. Чётко разделяй:
   - данные из графа;
   - собственные пояснения.

При анализе:
- найди основной объект вопроса;
- объясни его назначение;
- покажи связанные узлы;
- объясни значение связей.

Формат ответа:

## Основной объект

## Связи из БДУ

## Анализ

## Практическое применение

## Ограничения данных

"""

    prompt = f"""
Граф знаний:

{context}


Запрос:

{text}


Ответ:
"""

    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen",
            "messages": [
                {
                    "role": "system",
                    "content": system,
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

Отвечай точно и структурировано.
Не придумывай факты, если не уверен.
Если вопрос требует конкретных данных, укажи, что у тебя нет доступа к ним.

Структура ответа:
1. Краткое определение.
2. Принцип работы.
3. Возможные последствия.
4. Методы обнаружения.
5. Меры защиты.

Используй профессиональную терминологию.
"""

    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen",
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
    if response.status_code != 200:
        print(response.text)

    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)
