# cleaning, validation and normalization
from dataclasses import asdict
import json
import csv
import re
from pathlib import Path

import src.models as obj


class BDUParser:
    def __init__(self, config):
        self.cfg = config["parser"]

        self.files = config["files"]["nodes"]
        self.output = config["files"]["processed"]

        self.datasets = {}

        self.nodes: list[obj.Node] = []
        self.edges: list[obj.Edge] = []

        self.node_index: dict[tuple[str, str], obj.Node] = {}
        self.edge_index: set[tuple[int, int, str]] = set()

        # поиск по имени (для отладки и fallback)
        self.name_index: dict[tuple[str, str], obj.Node] = {}

        self.next_node_id = 0

    # =====================================================
    # PUBLIC API
    # =====================================================

    def run(self):
        self.load_files()

        print("[INFO] Collecting nodes...")
        self.collect_nodes()

        print("[INFO] Collecting edges...")
        self.collect_edges()

        print(f"[INFO] Nodes: {len(self.nodes)}")
        print(f"[INFO] Edges: {len(self.edges)}")

        return self.nodes, self.edges

    def save(self):
        with open(
            self.output["nodes"],
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                [asdict(node) for node in self.nodes],
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(
            self.output["edges"],
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                [asdict(edge) for edge in self.edges],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def load_files(self):
        for name, path in self.files.items():
            path = Path(path)

            if path.suffix == ".csv":
                self.datasets[name] = self._load_csv(path)

            elif path.suffix == ".json":
                self.datasets[name] = self._load_json(path)

            else:
                raise ValueError(f"Unknown file type: {path}")

    def collect_nodes(self):
        for node_type, cfg in self.cfg["nodes"].items():
            source = cfg["source"]

            dataset = self.datasets[source]
            if cfg["source_type"] == "csv":
                self._collect_csv_nodes(
                    node_type=node_type,
                    dataset=dataset,
                    cfg=cfg,
                )

            else:
                self._collect_json_nodes(
                    node_type=node_type,
                    dataset=dataset,
                    cfg=cfg,
                )

    def collect_edges(self):
        for rule in self.cfg["edges"]:
            print(f"[EDGE] {rule['from']} -> {rule['to']} : {rule['edge']}")

            dataset = self.datasets[rule["source"]]

            self._collect_csv_edges(
                dataset=dataset,
                cfg=rule,
            )

    # =====================================================
    # CSV
    # =====================================================

    def _collect_csv_nodes(
        self,
        node_type: str,
        dataset: list[dict],
        cfg: dict,
    ):
        """
        Создает вершины из CSV.

        cfg:
        {
            "source": "file_path",
            "internal_id": "",
            "name": "",
            "description": []
        }
        """
        id_column = cfg.get("internal_id", "")
        name_column = cfg["name"]
        description_columns = cfg.get("description", [])

        added = 0

        for row in dataset:
            raw_name = row.get(name_column, "").strip()

            if not raw_name:
                continue

            # Есть отдельная колонка ID
            if id_column:
                node_id = row.get(id_column, "").strip()
                name = raw_name

            # ID находится внутри имени
            else:
                parts = raw_name.split(maxsplit=1)

                if len(parts) < 2:
                    print(
                        "[ERROR] Cannot extract id\n",
                        f"type: {node_type}\n",
                        f"value: {raw_name}",
                    )
                    continue

                node_id = parts[0]
                name = parts[1]

            if not node_id:
                print("[ERROR] Empty node id", node_type, raw_name)
                continue
            description = []

            for column in description_columns:
                value = row.get(column, "").strip()

                if value:
                    description.append(value)

            _, created = self._add_node(
                node_type=node_type,
                node_id=node_id,
                name=name,
                description="\n".join(description),
            )

            if created:
                added += 1

        print(f"{node_type}: +{added} nodes")

    # =====================================================
    # JSON
    # =====================================================

    def _collect_json_nodes(
        self,
        node_type: str,
        dataset: dict,
        cfg: dict,
    ):
        """
        Создает вершины из JSON API.
        """

        section = cfg.get("section", "data")

        id_field = cfg.get("id", "id")
        name_field = cfg.get("name", "name")
        description_field = cfg.get("description")

        items = dataset.get(section, [])

        added = 0
        skipped = 0

        for item in items:
            attributes = item.get("attributes", {})

            external_id = item.get(id_field)

            if external_id is None:
                print(f"[ERROR] {node_type}: node without id:")
                print(item)
                skipped += 1
                continue

            name = attributes.get(name_field)

            if not name:
                print(f"[ERROR] {node_type}: node without name:")
                print(item)
                skipped += 1
                continue

            description = ""

            if description_field:
                description = attributes.get(description_field) or ""

            node_id = f"{external_id}"

            self._add_node(
                node_type=node_type,
                node_id=node_id,
                name=name,
                description=description,
            )

            added += 1

        print(f"{node_type}: +{added} nodes, skipped={skipped}")

    def _add_node(
        self,
        node_type: str,
        node_id: str,
        name: str,
        description: str = "",
    ):
        key = (node_type, node_id)

        node = self.node_index.get(key)

        if node is not None:
            return node, False

        self.next_node_id += 1

        node = obj.Node(
            id=self.next_node_id,
            type=node_type,
            internal_id=node_id,
            name=name,
            description=description,
        )

        self.node_index[key] = node
        self.name_index[(node_type, name.strip())] = node

        self.nodes.append(node)

        return node, True

    def _load_csv(
        self,
        path: Path,
    ) -> list[dict]:

        with path.open(encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def _load_json(
        self,
        path: Path,
    ) -> dict:

        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def _collect_csv_edges(
        self,
        dataset: list[dict],
        cfg: dict,
    ):
        extract = cfg["extract"]

        from_type = cfg["from"]
        to_type = cfg["to"]

        relation = cfg["edge"]

        from_column = extract["from"]
        to_column = extract["to"]

        from_type_parser = extract["from_type"]
        to_type_parser = extract["to_type"]

        added = 0
        errors = 0

        for row in dataset:
            source_keys = self._parse_field(
                row,
                from_column,
                from_type_parser,
            )

            target_keys = self._parse_field(
                row,
                to_column,
                to_type_parser,
            )

            for source_key in source_keys:
                source = self.node_index.get(
                    (
                        from_type,
                        source_key,
                    )
                )

                if source is None:
                    print("[ERROR] Source not found")
                    print("type:", from_type)
                    print("id:", repr(source_key))
                    errors += 1
                    continue

                for target_key in target_keys:
                    target = self.node_index.get(
                        (
                            to_type,
                            target_key,
                        )
                    )

                    if target is None:
                        target_by_name = self.name_index.get(
                            (
                                to_type,
                                target_key,
                            )
                        )

                        if target_by_name:
                            target = target_by_name

                    if target is None:
                        print("\n[ERROR] Target not found")
                        print("target type:", to_type)
                        print("searched id:", repr(target_key))
                        print("source:")
                        print(
                            " ",
                            source.type,
                            source.internal_id,
                            source.name,
                        )

                        errors += 1
                        continue

                    edge_key = (
                        source.id,
                        target.id,
                        relation,
                    )

                    if edge_key in self.edge_index:
                        continue

                    self.edge_index.add(edge_key)

                    self.edges.append(
                        obj.Edge(
                            source=source.id,
                            target=target.id,
                            relation=relation,
                        )
                    )

                    added += 1

        print(f"    edges +{added}, errors={errors}")

    def _build_key(self, row, columns):

        if isinstance(columns, list):
            return row.get(columns[0], "").strip()

        value = row.get(columns, "").strip()

        if not value:
            return ""

        return value.split(maxsplit=1)[0]

    def _split_values(self, row, columns):

        result = []

        # -----------------------------------------
        # В конфиге пришёл массив:
        #
        # [
        #   "Идентификатор",
        #   "Наименование"
        # ]
        #
        # значит ID уже отдельно
        # -----------------------------------------

        if isinstance(columns, list):
            value = row.get(columns[0], "").strip()

            if value:
                result.append(value)

            return result

        # -----------------------------------------
        # Обычная колонка:
        #
        # "УБИ.1 Название"
        #
        # надо достать ID
        # -----------------------------------------

        value = row.get(columns, "")

        if not value:
            return []

        for item in value.splitlines():
            item = item.strip()

            if not item:
                continue

            node_id = item.split(maxsplit=1)[0]

            if node_id in (
                "Данные",
                "уточняются",
                "Данные_уточняются",
            ):
                continue

            result.append(node_id)

        return result

    def _extract_id(self, value: str, split: bool):

        value = value.strip()

        if not value:
            return ""

        if not split:
            return value

        return value.split(maxsplit=1)[0]

    def _parse_field(
        self,
        row: dict,
        column,
        field_type: str,
    ) -> list[str]:

        value = row.get(column, "")

        if not value:
            return []

        value = value.strip()

        # ------------------------------
        # clear_str
        #
        # К.1.1.1
        # ------------------------------

        if field_type == "clear_str":
            return [value]

        # ------------------------------
        # mixed_str
        #
        # К.1 Название
        # ------------------------------
        if field_type == "mixed_str":
            node_id = value.split(maxsplit=1)[0]

            if node_id.lower() in (
                "данные",
                "уточняются",
            ):
                return []

            return [node_id]

        # ------------------------------
        # mixed_arr
        #
        # К.1 Название;
        # К.2 Название;
        #
        # ------------------------------

        if field_type == "mixed_arr":
            node_id = value.split(maxsplit=1)[0]

            if node_id.lower() in (
                "данные",
                "уточняются",
            ):
                return []
            result = []

            matches = re.findall(
                r"\b[А-ЯA-ZЁ]{1,}\.\d+(?:\.\d+)*",
                value,
            )

            if not matches:
                print("\n========== NO MATCH ==========")
                print("column:", column)
                print("type:", field_type)
                print("value:")
                print(value[:500])
                print("==============================\n")

            for node_id in matches:
                result.append(node_id)

            return list(dict.fromkeys(result))

        raise ValueError(f"Unknown field type: {field_type}")
