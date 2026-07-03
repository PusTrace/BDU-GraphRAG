# cleaning, validation and normalization

import json


PARSER_CONFIG = {
    "cmptypes": {
        "file": "data/cmptypes_list.json",
        "roots": [
            {"type": "cmptype", "source": "data"},
            {"type": "cmp", "source": "included"},
        ],
        "relations": [
            {
                "from": "cmptype",
                "relation": "components",
                "to": "cmp",
                "edge": "HAS_COMPONENT",
            },
            {
                "from": "cmp",
                "parent": "parent_id",
                "edge": "HAS_CHILD",
            },
        ],
    },
    "components": {
        "file": "data/components_list.json",
        "roots": [
            {"type": "cmp", "source": "data"},
            {"type": "defense", "source": "included"},
        ],
        "relations": [
            {
                "from": "cmp",
                "relation": "cmptype",
                "to": "cmptype",
                "edge": "BELONGS_TO_TYPE",
            },
            {
                "from": "cmp",
                "relation": "technics",
                "to": "technic",
                "edge": "HAS_TECHNIC",
            },
            {
                "from": "cmp",
                "parent": "parent_id",
                "edge": "HAS_CHILD",
            },
            {
                "from": "defense",
                "parent": "parent_id",
                "edge": "HAS_CHILD",
            },
        ],
    },
    "defenses": {
        "file": "data/defenses_list.json",
        "roots": [
            {
                "type": "defense",
                "source": "data",
            },
        ],
        "relations": [
            {
                "from": "defense",
                "parent": "parent_id",
                "edge": "HAS_CHILD",
            },
        ],
    },
    "defgroups": {
        "file": "data/defgroups_list.json",
        "roots": [
            {"type": "defgroup", "source": "data"},
            {"type": "defense", "source": "included"},
        ],
        "relations": [
            # defgroup -> defenses (из data.relationships.defenses)
            {
                "from": "defgroup",
                "relation": "defenses",
                "to": "defense",
                "edge": "HAS_DEFENSE",
            },
            # defense hierarchy (included.childs)
            {
                "from": "defense",
                "relation": "childs",
                "to": "defense",
                "edge": "HAS_CHILD",
            },
            # fallback hierarchy via parent_id (если childs неполный)
            {
                "from": "defense",
                "parent": "parent_id",
                "edge": "HAS_CHILD",
            },
        ],
    },
    "negatives": {
        "file": "data/negatives_list.json",
        "roots": [
            {"type": "negative", "source": "data"},
        ],
        "relations": [],
    },
    "objects": {
        "file": "data/objects_list.json",
        "roots": [
            {"type": "object", "source": "data"},
            {"type": "cmp", "source": "included"},
        ],
        "relations": [
            # object -> components
            {
                "from": "object",
                "relation": "components",
                "to": "cmp",
                "edge": "HAS_COMPONENT",
            },
            # cmp -> cmptype
            {
                "from": "cmp",
                "relation": "cmptype",
                "to": "cmptype",
                "edge": "BELONGS_TO_TYPE",
            },
            # cmp hierarchy
            {
                "from": "cmp",
                "parent": "parent_id",
                "edge": "HAS_CHILD",
            },
        ],
    },
    "potentials": {
        "file": "data/potentials_list.json",
        "roots": [
            {"type": "potential", "source": "data"},
            {"type": "intview", "source": "included"},
        ],
        "relations": [
            {
                "from": "potential",
                "relation": "intviews",
                "to": "intview",
                "edge": "HAS_INTVIEW",
            }
        ],
    },
    "techgroups": {
        "file": "data/techgroups_list.json",
        "roots": [
            {"type": "techgroup", "source": "data"},
            {"type": "technic", "source": "included"},
        ],
        "relations": [
            # techgroup -> technic
            {
                "from": "techgroup",
                "relation": "technics",
                "to": "technic",
                "edge": "HAS_TECHNIC",
            },
            # technic -> cmp (из included)
            {
                "from": "technic",
                "relation": "components",
                "to": "cmp",
                "edge": "AFFECTS_COMPONENT",
            },
        ],
    },
    "technics": {
        "file": "data/technics_list.json",
        "roots": [
            {"type": "technic", "source": "data"},
        ],
        "relations": [
            {
                "from": "technic",
                "fk": "pot_id",
                "to": "potential",
                "edge": "HAS_POTENTIAL",
            }
        ],
    },
    "threats": {
        "file": "data/threats_list.json",
        "roots": [
            {"type": "threat", "source": "data"},
            {"type": "technic", "source": "included"},
        ],
        "relations": [
            {
                "from": "threat",
                "relation": "technics",
                "to": "technic",
                "edge": "CAUSED_BY",
            },
            {
                "from": "technic",
                "relation": "techgroups",
                "to": "techgroup",
                "edge": "BELONGS_TO_GROUP",
            },
        ],
    },
}


class BDUParser:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.id_map = {}

    # =====================================================
    # PUBLIC API
    # =====================================================

    def parse(self, config: dict):
        path = config.get("file")
        if path is None:
            raise ValueError("config file path is none")
        self._reset()

        data = self._load(path)

        # -------- nodes --------
        roots = config["roots"]
        for root in roots:
            items = data.get(root["source"], [])
            print(f"roots: {len(items)}")

            for item in items:
                if item.get("type") != root["type"]:
                    continue

                self._add_node(item)

        # -------- relations --------
        relations = config["relations"]
        for rule in relations:
            self._build_relation(data, rule)

        return self.nodes, self.edges

    # =====================================================
    # PRIVATE
    # =====================================================

    def _reset(self):
        self.nodes = []
        self.edges = []
        self.id_map = {}

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


def main():
    parser = BDUParser()

    # nodes, edges = parser.parse(PARSER_CONFIG["negatives"])
    for key, target in PARSER_CONFIG.items():
        print(f"\n{key}")
        nodes, edges = parser.parse(target)
        print(f"cleaned nodes: {len(nodes)}")
        print(f"cleaned edges: {len(edges)}")

    # print(json.dumps(nodes, ensure_ascii=False, indent=2))
    # input("enter for cont")
    # print(json.dumps(edges, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
