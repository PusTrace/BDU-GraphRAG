# cleaning, validation and normalization
import json

import src.storage as load


class BDUParser:
    def __init__(self, config):
        self.cfg = config["parser"]
        self.files = config["files"]["raw"]

        self.data = {}

        self.id_map = {}
        self.nodes = []
        self.edges = []

    # =====================================================
    # PUBLIC API
    # =====================================================

    def load_all_files(self):
        print("[INFO] Loading datasets...")

        for name, file_path in self.files.items():
            print(f"Loading {name}: {file_path}")

            with open(file_path, encoding="utf8") as f:
                self.data[name] = json.load(f)

            print(
                f"{name}: "
                f"data={len(self.data[name].get('data', []))}, "
                f"included={len(self.data[name].get('included', []))}"
            )

        print(f"[INFO] Loaded {len(self.data)} datasets")

    def run(self):
        self.load_all_files()
        self.generate_index()
        self.collect_nodes()
        self.generate_network()
        print(f"[INFO] cleaned nodes: {len(self.nodes)}")
        print(f"[INFO] cleaned edges: {len(self.edges)}")
        return self.nodes, self.edges

    def generate_index(self):
        print("[INFO] Building global id index...")

        self.id_map = {}

        total = 0
        skipped = 0

        for dataset_name, dataset in self.data.items():
            before = len(self.id_map)

            for section in ("data", "included"):
                for item in dataset.get(section, []):
                    attrs = item.get("attributes", {})

                    identifier = attrs.get("identifier")

                    if not identifier:
                        skipped += 1
                        continue

                    try:
                        key = (
                            item["type"],
                            int(item["id"]),
                        )
                    except Exception:
                        skipped += 1
                        continue

                    self.id_map[key] = identifier
                    total += 1

            print(f"{dataset_name}: +{len(self.id_map) - before} ids")

        print(f"[INFO] Indexed {len(self.id_map)} unique ids")
        print(f"processed={total}, skipped={skipped}")

        return self.id_map

    def collect_nodes(self):
        print("[INFO] Collecting nodes...")

        self.nodes = []

        seen = set()

        duplicates = 0
        skipped = 0

        for dataset_name, dataset in self.data.items():
            before = len(self.nodes)

            for section in ("data", "included"):
                for item in dataset.get(section, []):
                    attrs = item.get("attributes", {})

                    identifier = attrs.get("identifier")

                    if not identifier:
                        skipped += 1
                        continue

                    if identifier in seen:
                        duplicates += 1
                        continue

                    seen.add(identifier)

                    self.nodes.append(
                        {
                            "id": identifier,
                            "type": item["type"],
                            "name": attrs.get("name"),
                        }
                    )

            print(f"{dataset_name}: +{len(self.nodes) - before} nodes")

        print(f"[INFO] Collected {len(self.nodes)} nodes")
        print(f"duplicates={duplicates}, skipped={skipped}")

        return self.nodes

    def generate_network(self):
        print("[INFO] Building graph...")

        self.edges = []

        seen = set()

        duplicates = 0
        missing_targets = 0
        missing_sources = 0

        for dataset_name, dataset in self.data.items():
            parser_cfg = self.cfg.get(dataset_name)

            if parser_cfg is None:
                print(f"{dataset_name}: no parser config")
                continue

            dataset_edges = 0

            for rule in parser_cfg.get("relations", []):
                print(f"{dataset_name}: {rule['from']} -> {rule['edge']}")

                before = len(self.edges)

                for section in ("data", "included"):
                    for item in dataset.get(section, []):
                        if item.get("type") != rule["from"]:
                            continue

                        source = self._id(
                            item["type"],
                            item.get("id"),
                        )

                        if not source:
                            missing_sources += 1
                            continue

                        # parent
                        if "parent" in rule:
                            parent_id = item.get(
                                "attributes",
                                {},
                            ).get(rule["parent"])

                            target = self._id(
                                item["type"],
                                parent_id,
                            )

                            if not target:
                                missing_targets += 1
                                continue

                            edge = (
                                target,
                                rule["edge"],
                                source,
                            )

                            if edge in seen:
                                duplicates += 1
                                continue

                            seen.add(edge)

                            self.edges.append(
                                {
                                    "source": target,
                                    "target": source,
                                    "relation": rule["edge"],
                                }
                            )

                            continue
                        if "fk" in rule:
                            fk = item.get("attributes", {}).get(rule["fk"])

                            target = self._id(
                                rule["to"],
                                fk,
                            )

                            if not target:
                                continue

                            edge = (
                                source,
                                rule["edge"],
                                target,
                            )

                            if edge in seen:
                                continue

                            seen.add(edge)

                            self.edges.append(
                                {
                                    "source": source,
                                    "target": target,
                                    "relation": rule["edge"],
                                }
                            )

                            continue

                        # normal relation
                        for rel in self._relation(
                            item,
                            rule["relation"],
                        ):
                            target = self._id(
                                rule["to"],
                                rel.get("id"),
                            )

                            if not target:
                                missing_targets += 1
                                continue

                            edge = (
                                source,
                                rule["edge"],
                                target,
                            )

                            if edge in seen:
                                duplicates += 1
                                continue

                            seen.add(edge)

                            self.edges.append(
                                {
                                    "source": source,
                                    "target": target,
                                    "relation": rule["edge"],
                                }
                            )

                added = len(self.edges) - before
                dataset_edges += added

                print(f"    added {added} edges")

            print(f"[INFO] {dataset_name}: {dataset_edges} edges")

        print("[INFO] Graph complete")
        print(f"[INFO] Total edges: {len(self.edges)}")

        print(f"duplicate edges: {duplicates}")
        print(f"missing source ids: {missing_sources}")
        print(f"missing target ids: {missing_targets}")

        return self.edges

    def parse(self):
        path = self.cfg.get("file")
        if path is None:
            raise ValueError("config file path is none")

        data = self._load(path)

        # -------- nodes --------
        roots = self.cfg["roots"]
        for root in roots:
            items = data.get(root["source"], [])
            print(f"roots: {len(items)}")

            for item in items:
                if item.get("type") != root["type"]:
                    continue

                self._add_node(item)

        # -------- relations --------
        relations = self.cfg["relations"]
        for rule in relations:
            self._build_relation(data, rule)

        return self.nodes, self.edges

    # =====================================================
    # PRIVATE
    # =====================================================

    def _load(self, path: str):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _add_node(self, item):
        attrs = item.get("attributes", {})

        internal_id = item.get("id")
        node_type = item.get("type")

        identifier = attrs.get("identifier")
        name = attrs.get("name")

        if not identifier:
            return

        try:
            key = (node_type, int(internal_id))
        except Exception:
            return

        self.id_map[key] = identifier

        self.nodes.append(
            {
                "id": identifier,
                "type": node_type,
                "name": name,
            }
        )

    def _relation(self, item, relation):
        data = item.get("relationships", {}).get(relation, {}).get("data")

        if not data:
            return []

        if isinstance(data, dict):
            return [data]

        if isinstance(data, list):
            return data

        return []

    def _id(self, node_type, internal_id):
        if internal_id is None:
            return None

        try:
            return self.id_map.get((node_type, int(internal_id)))
        except Exception:
            return None

    def _edge(self, source, target, relation):
        if not source or not target:
            return

        self.edges.append(
            {
                "source": source,
                "target": target,
                "relation": relation,
            }
        )

    def _build_relation(self, data, rule):

        for section in ("data", "included"):
            for item in data.get(section, []):
                if item.get("type") != rule["from"]:
                    continue

                source = self._id(item["type"], item.get("id"))

                if not source:
                    continue

                # -------- parent_id relations --------
                if "parent" in rule:
                    parent_id = item.get("attributes", {}).get(rule["parent"])

                    parent = self._id(item["type"], parent_id)

                    self._edge(parent, source, rule["edge"])
                    continue

                # -------- normal relationships --------
                for rel in self._relation(item, rule.get("relation")):
                    target = self._id(rule["to"], rel.get("id"))

                    self._edge(source, target, rule["edge"])
