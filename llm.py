# llm

import requests


def llm(user_input, context):
    prompt = f"""
Ты ассистент подключённый к базе знаний БДУ ФСТЕК(https://bdu.fstec.ru).
Используй только предоставленный контекст когда контекст валиден к ответу.

Контекст:
{context}

Вопрос:
{user_input}

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


def llm_without_graph(user_input):
    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": user_input}],
        },
    )

    return response
