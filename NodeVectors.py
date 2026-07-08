import requests
import json


def calc_embedding(sentence):
    response = requests.post(
        "http://localhost:8081/embedding",
        json={"content": sentence},
    )

    return response.json()[0]["embedding"][0]


def main():
    with open("data/nodes.json", "r", encoding="utf-8") as f:
        nodes = json.load(f)

    result = []

    for node in nodes:
        node_id = node["id"]
        node_type = node["type"]
        name = node["name"]

        text = f"{node_id} ({node_type}) - {name}"

        print("embedding:", text)

        embedding = calc_embedding(text)

        result.append(
            {
                "id": node_id,
                "text": text,
                "embedding": embedding,
            }
        )

    with open("data/node_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

