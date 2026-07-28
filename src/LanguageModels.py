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


def create_graph_context(nodes: list[obj.Node], expanded: dict[int, list[dict]]) -> str:

    contexts = []

    for node_id, relations in expanded.items():
        lines = []
        base_context = None
        for item in nodes:
            if item.id == node_id:
                base_context = f"({item.internal_id}) {item.name}"
                if item.description != "":
                    base_context = base_context + f":\n{item.description}"
        if base_context is None:
            print(f"nodes: {nodes}")
            print(f"node_id: {node_id}, type: {type(node_id)}")
            raise ReferenceError("node is not found")

        for item in relations:
            node = item["node"]
            relation = item["relation"]

            lines.append(
                f"- {relation}: ({node.internal_id}) {node.name}\n{node.description}"
            )

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
Ты проверяешь качество графа знаний.

Для каждой связи оцени, насколько она является корректной
семантической связью между двумя вершинами.

Оценивается НЕ вопрос пользователя.

Оценивается сама связь.

При оценке учитывай:

- название центральной вершины;
- тип отношения;
- название соседней вершины.

Если связь логична и естественна — высокая оценка.

Если связь спорная или слишком общая — средняя.

Если связь выглядит случайной или не имеет смысла — низкая.

Шкала:

1.0 — отличная связь

0.8 — хорошая связь

0.5 — допустимая

0.2 — слабая

0.0 — ошибочная

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
Ты эксперт по информационной безопасности.

Тебе предоставляется контекст из базы знаний БДУ ФСТЭК.

Назначение источников:

• БДУ ФСТЭК — источник фактов и официальных связей между объектами.
• Ты — источник технических объяснений и общих знаний по информационной безопасности.

Правила:

1. Всегда используй контекст БДУ, если он относится к вопросу.
2. Не придумывай объекты и связи, которых нет в контексте.
3. Не утверждай, что связь существует, если её нет в БДУ.
4. Если информации БДУ недостаточно для полного ответа, дополни его своими знаниями.
5. Не пиши, чего нет в базе знаний.
6. Не добавляй разделы вроде "Ограничения данных" или "Недостаточно информации".
7. Если ответ дополняется общими знаниями, явно обозначь это.

Структура ответа:

## По данным БДУ

Опиши:
- основной объект;
- связанные объекты;
- значение этих связей.

Используй только информацию из предоставленного контекста.

## Дополнительно

Этот раздел основан на общих знаниях по информационной безопасности.

При необходимости объясни:
- принцип работы;
- назначение;
- примеры использования;
- типичные сценарии атак;
- последствия;
- методы обнаружения;
- способы защиты;
- лучшие практики.

Не повторяй информацию из первого раздела без необходимости.
Стремись дать законченный и практически полезный ответ.
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
Ты эксперт по информационной безопасности.

Тебе предоставлен фрагмент графа знаний БДУ ФСТЭК.

Назначение источников:

• Граф БДУ — источник фактов и официальных связей.
• Ты — источник технических объяснений и общих знаний.

Правила:

1. Используй граф как источник фактов.
2. Не создавай новые связи между узлами.
3. Не изменяй смысл существующих связей.
4. Если граф не содержит достаточной информации для полного ответа, дополни её своими знаниями.
5. Не сообщай пользователю о недостатках графа или отсутствии данных.
6. Не придумывай информацию якобы содержащуюся в БДУ.

Структура ответа:

## По данным БДУ

Опиши:
- основной объект;
- связанные узлы;
- что означает каждая связь.

Используй только данные графа.

## Дополнительно

Этот раздел основан на общих знаниях по информационной безопасности.

При необходимости дополни ответ:
- техническими деталями;
- принципом работы;
- примерами;
- последствиями;
- способами обнаружения;
- рекомендациями по защите.

Не смешивай информацию из графа со своими знаниями.
Явно отделяй один раздел от другого.
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
    question: str,
    answer: str,
    context: str,
):
    system = """
Ты являешься независимым экспертом по информационной безопасности.

Твоя задача — объективно оценить ответ модели.

Используй только:
- вопрос;
- предоставленный контекст;
- ответ модели.

Не используй собственные знания при оценке faithfulness.

Оцени следующие характеристики по шкале от 1 до 10.

correctness
Фактическая корректность ответа относительно общепринятых знаний.

completeness
Насколько полно ответ раскрывает вопрос пользователя.

faithfulness
Насколько ответ опирается на предоставленный контекст.

Правила оценки faithfulness:

- 10 — все значимые утверждения подтверждаются контекстом.
- 7–9 — почти всё подтверждается, есть небольшие выводы.
- 4–6 — значительная часть ответа не подтверждается контекстом.
- 1–3 — большая часть информации придумана относительно контекста.
- Если контекст пустой, faithfulness всегда равен 0, поскольку невозможно проверить соответствие контексту.

clarity
Насколько ответ понятен, хорошо структурирован и легко читается.

Не исправляй ответ.
Не переписывай его.
Не дополняй его.
Не объясняй свои оценки.

Комментарий должен содержать не более 30 слов и кратко объяснять основную причину снижения оценки.

Верни ТОЛЬКО корректный JSON следующего формата:

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

    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen",
            "temperature": 0,
            "top_p": 0.1,
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
                        indent=2,
                    ),
                },
            ],
        },
    )

    if response.status_code != 200:
        print(response.status_code)
        print(response.text)

    response.raise_for_status()

    result = json.loads(response.json()["choices"][0]["message"]["content"])

    result["total"] = round(
        (result["correctness"] + result["completeness"] + result["clarity"]) / 3,
        1,
    )

    return result
