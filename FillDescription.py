import json
import requests
from pathlib import Path

NODES_FILE = Path("data/processed/nodes.json")


def generate_description(name: str, node_type: str) -> str:
    system = """
Ты эксперт по информационной безопасности.

Тебе дан объект базы знаний БДУ ФСТЭК.

Напиши определение длиной 20–80 слов.

Правила:

- Объясни назначение объекта.
- Не упоминай БДУ.
- Не перечисляй связанные объекты.
- Не описывай меры защиты.
- Не используй списки.
- Пиши энциклопедическим стилем.
- Не начинай с "Это..." если можно написать лучше.
- Верни только описание без заголовков и форматирования.

"""

    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen",
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "name": name,
                            "type": node_type,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"].strip()


def main():
    with NODES_FILE.open(encoding="utf-8") as f:
        nodes = json.load(f)

    total = len(nodes)

    for i, node in enumerate(nodes, start=1):
        description = node.get("description", "").strip()

        if description:
            continue

        print(f"[{i}/{total}] {node['internal_id']} {node['name']}")

        try:
            node["description"] = generate_description(
                node["name"],
                node["type"],
            )

            print("✓ Сгенерировано")

            # сохраняем сразу после каждой успешной генерации
            with NODES_FILE.open("w", encoding="utf-8") as f:
                json.dump(
                    nodes,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        except Exception as e:
            print(f"✗ Ошибка: {e}")

    print("Готово.")


if __name__ == "__main__":
    main()
