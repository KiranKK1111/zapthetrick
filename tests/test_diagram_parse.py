"""Mermaid → IR lift, and the round trip back out.

This is what lets an EXISTING diagram (model-written or hand-edited) join the IR
pipeline for validation, editing, versioning and export. Two properties matter:
nothing is silently lost (whatever the reader can't interpret lands in
`meta["unparsed"]`), and the re-emitted source still parses.

Every fixture below was round-tripped through real mermaid 11.15.0 during
development; the counts asserted here are the measured, correct ones.
"""
from __future__ import annotations

import app.diagrams.parse as P
from app.diagrams.ir import to_mermaid


def test_flowchart_basic():
    ir = P.from_mermaid("""flowchart LR
    User --> API
    API --> Auth
    API --> LLM
    LLM --> Database
    Database --> API
""")
    assert ir.kind == "flowchart" and ir.direction == "LR"
    assert ir.node_ids == {"User", "API", "Auth", "LLM", "Database"}
    assert len(ir.edges) == 5


def test_declarations_and_link_on_one_line_are_both_read():
    # The regression that motivated `_reduce_declarations`: a line mixing shapes
    # used to yield only the LAST shape family and no edge at all.
    ir = P.from_mermaid("flowchart TD\n  A[API] --> B[(DB)]")
    assert ir.node_ids == {"A", "B"}
    assert ir.node("A").shape == "rect"
    assert ir.node("B").shape == "cylinder"
    assert len(ir.edges) == 1


def test_pipe_labels_are_read():
    ir = P.from_mermaid("""graph TD
  A[Start] -->|submit| B{Valid?}
  B -->|yes| C[Process]
  B -->|no| D[Reject]
  C --> E([Done])
  D --> E
""")
    assert len(ir.edges) == 5
    labels = {e.label for e in ir.edges}
    assert {"submit", "yes", "no"} <= labels
    assert ir.node("B").shape == "rhombus"
    assert ir.node("E").shape == "stadium"


def test_middle_labels_and_every_link_style():
    ir = P.from_mermaid("""flowchart LR
  A[One] --- B[Two]
  B -.-> C[Three]
  C ==> D[Four]
  D -. retry .-> A
""")
    assert len(ir.edges) == 4
    by_pair = {(e.src, e.dst): e for e in ir.edges}
    assert by_pair[("A", "B")].arrow == "open"
    assert by_pair[("B", "C")].style == "dotted"
    assert by_pair[("C", "D")].style == "thick"
    assert by_pair[("D", "A")].label == "retry"
    assert by_pair[("D", "A")].style == "dotted"


def test_long_arrows_are_read_not_rejected():
    # `--->` is VALID mermaid (extra dashes set rank distance).
    ir = P.from_mermaid("flowchart LR\n  A ---> B\n  B ----> C")
    assert len(ir.edges) == 2


def test_chained_links_on_one_line():
    ir = P.from_mermaid("flowchart LR\n  A --> B --> C -.-> D")
    assert len(ir.edges) == 3
    assert ir.node_ids == {"A", "B", "C", "D"}


def test_bidirectional_link():
    ir = P.from_mermaid("flowchart LR\n  A <--> B")
    assert ir.edges[0].arrow == "bidirectional"


def test_subgraphs_nest_and_assign_membership():
    ir = P.from_mermaid("""flowchart TB
  subgraph web[Web tier]
    LB[Load balancer]
    W1[Worker 1]
  end
  subgraph data[Data tier]
    PG[(Postgres)]
  end
  LB --> W1
  W1 --> PG
""")
    assert {g.id for g in ir.groups} == {"web", "data"}
    assert ir.node("LB").group == "web"
    assert ir.node("PG").group == "data"
    assert ir.group("web").label == "Web tier"


def test_init_directive_and_comments_are_skipped():
    ir = P.from_mermaid("""%%{init: {"theme":"dark"}}%%
flowchart LR
  %% a comment
  X --> Y
""")
    assert len(ir.edges) == 1
    assert not ir.meta.get("unparsed")


def test_accessibility_metadata_is_lifted():
    ir = P.from_mermaid("""flowchart TD
  accTitle: Request path
  accDescr: A request goes to the API and then the database.
  A[API] --> B[(DB)]
""")
    assert ir.acc_title == "Request path"
    assert ir.acc_descr.startswith("A request goes")
    assert len(ir.edges) == 1


def test_acc_descr_block_form():
    ir = P.from_mermaid("""flowchart TD
  accDescr {
    line one
    line two
  }
  A --> B
""")
    assert ir.acc_descr == "line one\nline two"


def test_quoted_labels_are_unescaped():
    ir = P.from_mermaid("""flowchart TD
  A["Fetch (REST)"] --> B["Parse #quot;body#quot;"]
  B --> C["100#35;"]
""")
    assert ir.node("A").label == "Fetch (REST)"
    assert ir.node("B").label == 'Parse "body"'
    assert ir.node("C").label == "100#"


def test_sequence_participants_and_messages():
    ir = P.from_mermaid("""sequenceDiagram
    participant U as User
    participant S as Server
    U->>S: GET /items
    S-->>U: 200 OK
    Note over S: cached
""")
    assert ir.kind == "sequence"
    assert ir.node("U").label == "User"
    assert len(ir.edges) == 2
    assert ir.edges[1].style == "dotted"


def test_state_star_transitions_become_roles():
    ir = P.from_mermaid("""stateDiagram-v2
    [*] --> Idle
    Idle --> Busy : work arrives
    Busy --> [*] : shutdown
""")
    assert ir.kind == "state"
    roles = {n.id: n.role for n in ir.nodes}
    assert roles.get("start") == "start"
    assert roles.get("done") == "end"
    assert any(e.label == "work arrives" for e in ir.edges)


def test_er_relationships_and_cardinality():
    ir = P.from_mermaid("""erDiagram
    CUSTOMER ||--o{ ORDER : places
    ORDER ||--|{ LINE_ITEM : contains
""")
    assert ir.kind == "er"
    assert len(ir.edges) == 2
    assert ir.edges[0].cardinality == "1-*"
    assert ir.edges[0].label == "places"


def test_class_relations_and_members():
    ir = P.from_mermaid("""classDiagram
    Animal <|-- Duck
    Animal : +int age
    Animal : +isMammal()
    class Bird {
      +fly()
    }
""")
    assert ir.kind == "class"
    assert ir.edges[0].relation == "inheritance"
    assert ir.node("Animal").members == ["+int age", "+isMammal()"]
    assert ir.node("Bird").members == ["+fly()"]
    assert not ir.meta.get("unparsed")


def test_unreadable_lines_are_reported_not_dropped_silently():
    ir = P.from_mermaid("flowchart TD\n  A --> B\n  %%weird\n  ???!!!")
    assert "???!!!" in (ir.meta.get("unparsed") or [])


def test_fenced_source_is_unwrapped():
    ir = P.from_mermaid("```mermaid\nflowchart LR\nA --> B\n```")
    assert len(ir.edges) == 1


def test_from_mermaid_never_raises():
    for bad in ("", None, "```mermaid\n```", "flowchart", "\x00\x01"):
        assert P.from_mermaid(bad) is not None  # type: ignore[arg-type]


def test_round_trip_preserves_structure():
    source = """flowchart LR
  subgraph tier[Tier one]
    A["Fetch (REST)"] -->|ok| B[(Store)]
  end
  B -.-> C{Retry?}
"""
    first = P.from_mermaid(source)
    second = P.from_mermaid(to_mermaid(first))
    assert second.node_ids == first.node_ids
    assert len(second.edges) == len(first.edges)
    assert {g.id for g in second.groups} == {g.id for g in first.groups}
    assert second.node("A").label == "Fetch (REST)"


def test_round_trip_helper_returns_both_halves():
    ir, source = P.round_trip("flowchart LR\nA --> B")
    assert ir.node_ids == {"A", "B"}
    assert source.startswith("flowchart")


# ---- refusing what the IR does not model --------------------------------
# This is a SAFETY property, not a nicety: anything that re-emits from the IR
# (the answer-path compile lane, /api/diagram/normalize) would destroy a gantt
# chart that had been misread as a flowchart.
UNMODELLED_SOURCES = {
    "gantt": "gantt\n  dateFormat YYYY-MM-DD\n  section S\n  Task :a1, 2024-01-01, 30d",
    "pie": 'pie title Pets\n  "Dogs" : 386',
    "journey": "journey\n  title My day\n  section Work\n    Tea: 5: Me",
    "timeline": "timeline\n  title History\n  2002 : LinkedIn",
    "gitgraph": "gitGraph\n  commit\n  branch dev",
    "quadrantchart": "quadrantChart\n  title Reach\n  x-axis Low --> High",
    "sankey-beta": "sankey-beta\n\nA,B,10",
    "xychart-beta": 'xychart-beta\n  title "Sales"\n  bar [5, 10]',
    "block-beta": "block-beta\n  columns 1\n  A",
    "packet-beta": "packet-beta\n  0-15: header",
    "radar-beta": "radar-beta\n  axis a, b",
    "treemap": "treemap\n  Root\n    Child",
    "requirementdiagram": "requirementDiagram\n  requirement r {\n  id: 1\n  }",
    "kanban": "kanban\n  Todo\n    task1[Do it]",
    "architecture-beta": "architecture-beta\n  group api(cloud)[API]",
    "c4context": 'C4Context\n  title System\n  Person(a, "A")',
    "zenuml": "zenuml\n  A->B: hi",
}


def test_unmodelled_types_are_refused_not_guessed():
    for name, source in UNMODELLED_SOURCES.items():
        ir = P.from_mermaid(source)
        assert ir.nodes == [], name
        assert ir.meta.get("unsupported_kind") == name.lower(), name


def test_unsupported_kind_helper():
    assert P.unsupported_kind("gantt\n  title x") == "gantt"
    assert P.unsupported_kind("  pie title Pets") == "pie"
    # Past comments and an init directive, like mermaid's own type detection.
    assert P.unsupported_kind(
        "%% note\n%%{init: {}}%%\ntimeline\n  2002 : x") == "timeline"
    # Fenced input is unwrapped first.
    assert P.unsupported_kind("```mermaid\ngitGraph\n  commit\n```") == "gitgraph"
    # Modelled kinds and junk both return "".
    assert P.unsupported_kind("flowchart LR\n A --> B") == ""
    assert P.unsupported_kind("sequenceDiagram\n A->>B: hi") == ""
    assert P.unsupported_kind("") == ""
    assert P.unsupported_kind(None) == ""      # type: ignore[arg-type]


def test_a_node_label_containing_a_type_name_is_not_a_false_refusal():
    # "pie" as a LABEL must not be mistaken for a pie chart.
    ir = P.from_mermaid('flowchart LR\n  A["pie chart"] --> B["gantt"]')
    assert not ir.meta.get("unsupported_kind")
    assert ir.node_ids == {"A", "B"}


def test_flowchart_elk_header_is_read_as_a_flowchart():
    ir = P.from_mermaid("flowchart-elk LR\n  A --> B")
    assert ir.kind == "flowchart"
    assert len(ir.edges) == 1
