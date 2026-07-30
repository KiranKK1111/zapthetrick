"""Diagram IR + the deterministic Mermaid emitter (MermaidDiagramVisualizations.md #1).

The emitter's whole claim is "a well-formed IR cannot produce a diagram that fails
to parse". Every link spelling and node shape asserted here was verified against
the REAL mermaid 11.15.0 grammar (the version `zapthetrick_fe/assets/mermaid/
mermaid.min.js` bundles) with `mermaid.parse()`, so these are golden values, not
guesses. If a future change alters one of them, mermaid must be re-consulted
before the expectation is edited.
"""
from __future__ import annotations

import app.diagrams.ir as M


# ---- identifiers + labels -------------------------------------------------
def test_safe_id_folds_illegal_characters():
    assert M.safe_id("Kafka Broker 2") == "Kafka_Broker_2"
    assert M.safe_id("api-gw") == "api_gw"
    assert M.safe_id("café ☕") == "caf"


def test_safe_id_never_starts_with_a_digit():
    assert M.safe_id("9lives").startswith("n_")


def test_safe_id_falls_back_when_nothing_survives():
    assert M.safe_id("☕☕") == "n"
    assert M.safe_id("") == "n"


def test_escape_label_only_touches_quote_and_hash():
    # Measured: `:`, `<`, `>`, `;`, `&`, `%` are safe inside a quoted label;
    # `"` ends the string and `#` starts an entity, so only those two are escaped.
    assert M.escape_label('a "b" #1') == "a #quot;b#quot; #35;1"
    assert M.escape_label("a:b<c>d;e&f%g") == "a:b<c>d;e&f%g"


def test_escape_label_converts_newlines_to_br():
    # A quoted mermaid label may NOT span source lines. A planner asking for
    # two lines the natural way ("Broker 1\n:9092") used to emit a literal
    # newline inside the quotes → unparseable source, defeating the whole
    # point of the IR. Verified against the real engine: `<br/>` parses.
    assert M.escape_label("Broker 1\n:9092") == "Broker 1<br/>:9092"
    assert M.escape_label("a\r\nb") == "a<br/>b"
    assert M.escape_label("a\rb") == "a<br/>b"


def test_emitted_label_never_contains_a_raw_newline():
    ir = M.from_dict({
        "kind": "flowchart",
        "nodes": [{"id": "b1", "label": "Broker 1\n:9092"}],
        "edges": [],
    })
    body = M.to_mermaid(ir)
    # Every line of the emitted source must be a complete statement; a raw
    # newline inside a label would split one across two lines.
    assert "\n:9092" not in body
    assert 'b1["Broker 1<br/>:9092"]' in body


def test_wrap_label_breaks_on_words_and_is_idempotent():
    wrapped = M.wrap_label("one two three four five", 9)
    assert "<br/>" in wrapped
    assert M.wrap_label(wrapped, 9) == wrapped
    assert M.wrap_label("short", 9) == "short"


# ---- from_dict coercion --------------------------------------------------
def test_from_dict_aliases_edges_to_sanitised_ids():
    # An edge written against the ORIGINAL id must still resolve after the id is
    # folded — otherwise sanitising silently deletes structure.
    ir = M.from_dict({
        "nodes": [{"id": "api-gw", "label": "API"}, {"id": "DB", "label": "DB"}],
        "edges": [{"src": "api-gw", "dst": "DB"}],
    })
    assert ir.node_ids == {"api_gw", "DB"}
    assert (ir.edges[0].src, ir.edges[0].dst) == ("api_gw", "DB")


def test_from_dict_drops_unknown_values_not_the_diagram():
    ir = M.from_dict({
        "kind": "nonsense", "direction": "SIDEWAYS",
        "nodes": [{"id": "A", "label": "A", "shape": "octagon"}],
        "edges": [{"src": "A", "dst": "A", "style": "wavy", "arrow": "spiky"}],
    })
    assert ir.kind == "flowchart" and ir.direction == "TD"
    assert ir.nodes[0].shape == ""          # unknown shape → derived default
    assert ir.edges[0].style == "solid" and ir.edges[0].arrow == "arrow"


def test_from_dict_accepts_graph_alias_and_from_to_keys():
    ir = M.from_dict({
        "kind": "graph",
        "nodes": ["A", "B"],
        "edges": [{"from": "A", "to": "B"}],
    })
    assert ir.kind == "flowchart"
    assert len(ir.edges) == 1


def test_from_dict_deduplicates_nodes_and_rejects_bad_groups():
    ir = M.from_dict({
        "groups": [{"id": "g1", "label": "Tier"}],
        "nodes": [{"id": "A", "label": "one", "group": "g1"},
                  {"id": "A", "label": "again"},
                  {"id": "B", "label": "two", "group": "nope"}],
    })
    assert len(ir.nodes) == 2
    assert ir.node("A").group == "g1"
    assert ir.node("B").group == ""         # unknown group is not invented


def test_from_dict_never_raises():
    for bad in (None, [], "flowchart", {"nodes": "not a list"}, {"nodes": [1, 2]}):
        assert isinstance(M.from_dict(bad), M.DiagramIR)  # type: ignore[arg-type]


# ---- the emitter ---------------------------------------------------------
def _flow(**kwargs) -> M.DiagramIR:
    base = {"kind": "flowchart",
            "nodes": [{"id": "A", "label": "A"}, {"id": "B", "label": "B"}]}
    base.update(kwargs)
    return M.from_dict(base)


# Golden link spellings — each one confirmed by mermaid.parse(). `_LINKS_PLAIN`
# and `_LINKS_LABELLED` exist because mermaid's spellings are irregular: an
# arrowless solid link needs THREE dashes, and a dotted link's tail differs from
# its head.
LINK_GOLDEN = {
    ("solid", "arrow"): "-->", ("solid", "open"): "---",
    ("solid", "bidirectional"): "<-->",
    ("dotted", "arrow"): "-.->", ("dotted", "open"): "-.-",
    ("dotted", "bidirectional"): "<-.->",
    ("thick", "arrow"): "==>", ("thick", "open"): "===",
    ("thick", "bidirectional"): "<==>",
}


def test_every_link_spelling_is_valid():
    for (style, arrow), expected in LINK_GOLDEN.items():
        ir = _flow(edges=[{"src": "A", "dst": "B", "style": style, "arrow": arrow}])
        assert f"A {expected} B" in M.to_mermaid(ir), f"{style}/{arrow}"


def test_arrowless_solid_link_uses_three_dashes():
    # `A -- B` is a PARSE ERROR in mermaid; the open link is `A --- B`.
    out = M.to_mermaid(_flow(edges=[{"src": "A", "dst": "B", "arrow": "open"}]))
    assert "A --- B" in out
    assert "A -- B" not in out


def test_labelled_links_use_the_middle_form_with_quotes():
    out = M.to_mermaid(_flow(
        edges=[{"src": "A", "dst": "B", "label": "why (not)?"}]))
    assert 'A -- "why (not)?" --> B' in out


def test_labelled_dotted_link_tail_differs_from_head():
    out = M.to_mermaid(_flow(
        edges=[{"src": "A", "dst": "B", "label": "retry", "style": "dotted"}]))
    assert 'A -. "retry" .-> B' in out


def test_invisible_link_carries_no_label():
    out = M.to_mermaid(_flow(
        edges=[{"src": "A", "dst": "B", "label": "hidden", "style": "invisible"}]))
    assert "A ~~~ B" in out
    assert "hidden" not in out


def test_every_shape_delimiter_pair_is_emitted():
    for shape, (open_ch, close_ch) in M.SHAPES.items():
        ir = M.from_dict({"nodes": [{"id": "N", "label": "x", "shape": shape}]})
        assert f'N{open_ch}"x"{close_ch}' in M.to_mermaid(ir), shape


def test_role_picks_the_shape_when_none_is_given():
    ir = M.from_dict({"nodes": [
        {"id": "U", "label": "User", "role": "user"},
        {"id": "D", "label": "DB", "role": "datastore"},
        {"id": "Q", "label": "Q", "role": "decision"},
    ]})
    out = M.to_mermaid(ir)
    assert 'U("User")' in out               # actor → round
    assert 'D[("DB")]' in out               # datastore → cylinder
    assert 'Q{"Q"}' in out                  # decision → rhombus


def test_labels_are_always_quoted_and_escaped():
    ir = M.from_dict({"nodes": [{"id": "A", "label": 'Fetch (REST) "v2" #1'}]})
    assert 'A["Fetch (REST) #quot;v2#quot; #35;1"]' in M.to_mermaid(ir)


def test_long_labels_wrap():
    ir = M.from_dict({
        "label_wrap": 12,
        "nodes": [{"id": "A", "label": "a very long label indeed"}]})
    assert "<br/>" in M.to_mermaid(ir)


def test_subgraphs_are_always_balanced():
    ir = M.from_dict({
        "groups": [{"id": "outer", "label": "Outer"},
                   {"id": "inner", "label": "Inner", "parent": "outer"}],
        "nodes": [{"id": "A", "label": "A", "group": "outer"},
                  {"id": "B", "label": "B", "group": "inner"}],
    })
    out = M.to_mermaid(ir)
    assert out.count("subgraph ") == out.count("\n  end") + out.count("\n    end")
    # `end` must appear once per subgraph — balance is structural, not luck.
    assert len([line for line in out.splitlines()
                if line.strip() == "end"]) == 2


def test_sequence_diagram_never_gets_a_direction_token():
    # `sequenceDiagram TD` is a parse error — the emitter must not be able to.
    ir = M.from_dict({"kind": "sequence", "direction": "LR",
                      "nodes": [{"id": "A", "label": "A"}]})
    first = M.to_mermaid(ir).splitlines()[0]
    assert first == "sequenceDiagram"


def test_state_diagram_maps_start_and_end_roles_to_star():
    ir = M.from_dict({
        "kind": "state",
        "nodes": [{"id": "s", "label": "s", "role": "start"},
                  {"id": "Work", "label": "Work"},
                  {"id": "e", "label": "e", "role": "end"}],
        "edges": [{"src": "s", "dst": "Work"}, {"src": "Work", "dst": "e"}],
    })
    out = M.to_mermaid(ir)
    assert "[*] --> Work" in out
    assert "Work --> [*]" in out


def test_er_cardinality_maps_to_a_relationship_token():
    ir = M.from_dict({
        "kind": "er",
        "nodes": [{"id": "A", "label": "A"}, {"id": "B", "label": "B"}],
        "edges": [{"src": "A", "dst": "B", "label": "has", "cardinality": "1-*"}],
    })
    assert "A ||--o{ B : has" in M.to_mermaid(ir)


def test_class_relations_map_to_operators():
    ir = M.from_dict({
        "kind": "class",
        "nodes": [{"id": "Base", "label": "Base"}, {"id": "Sub", "label": "Sub"}],
        "edges": [{"src": "Base", "dst": "Sub", "relation": "inheritance"}],
    })
    assert "Base <|-- Sub" in M.to_mermaid(ir)


def test_accessibility_metadata_is_emitted():
    ir = M.from_dict({
        "nodes": [{"id": "A", "label": "A"}],
        "acc_title": "Flow", "acc_descr": "One box."})
    out = M.to_mermaid(ir)
    assert "accTitle: Flow" in out
    assert "accDescr: One box." in out


def test_multiline_acc_descr_uses_the_block_form():
    ir = M.from_dict({"nodes": [{"id": "A", "label": "A"}],
                      "acc_descr": "line one\nline two"})
    out = M.to_mermaid(ir)
    assert "accDescr {" in out and "\n  }" in out


def test_emitter_is_deterministic():
    payload = {"nodes": [{"id": "A", "label": "A"}, {"id": "B", "label": "B"}],
               "edges": [{"src": "A", "dst": "B", "label": "x"}]}
    assert M.to_mermaid(M.from_dict(payload)) == M.to_mermaid(M.from_dict(payload))


def test_empty_ir_still_emits_a_valid_diagram():
    out = M.to_mermaid(M.DiagramIR())
    assert out.startswith("flowchart TD")


def test_init_directive_is_prepended_verbatim():
    directive = '%%{init: {"theme":"dark"}}%%'
    out = M.to_mermaid(_flow(), init_directive=directive)
    assert out.splitlines()[0] == directive


def test_json_schema_matches_the_ir_vocabulary():
    schema = M.json_schema()
    props = schema["properties"]
    assert set(props["kind"]["enum"]) == set(M.KINDS)
    assert set(props["nodes"]["items"]["properties"]["shape"]["enum"]) == set(M.SHAPES)
    edge_props = props["edges"]["items"]["properties"]
    assert set(edge_props["style"]["enum"]) == set(M.EDGE_STYLES)
    assert set(edge_props["arrow"]["enum"]) == set(M.ARROWS)


def test_subgraph_titles_are_never_wrapped():
    # Mermaid reserves ONE line of height for a cluster title, so a `<br/>` in it
    # makes the title paint down into the cluster and over its first child node.
    ir = M.from_dict({
        "label_wrap": 12,
        "groups": [{"id": "unit",
                    "label": "Single Deployable Unit (Monolith)"}],
        "nodes": [{"id": "P", "label": "Presentation Layer (Controllers)",
                   "group": "unit"}],
    })
    out = M.to_mermaid(ir)
    title_line = next(line for line in out.splitlines() if "subgraph" in line)
    assert "<br/>" not in title_line
    assert 'Single Deployable Unit (Monolith)' in title_line
    # The NODE label inside it still wraps — only the title is exempt.
    node_line = next(line for line in out.splitlines() if line.strip().startswith("P["))
    assert "<br/>" in node_line
