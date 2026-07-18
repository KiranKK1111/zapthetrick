"""Phase 4 #3 — self-registering capability registry + contracts."""
from __future__ import annotations

from app.capabilities import registry as reg


def test_builtin_contracts_registered_in_snapshot():
    snap = reg.refresh()
    names = {c["name"] for c in snap.get("contracts", [])}
    assert {"document_render", "code_execution", "doc_transform"} <= names


def test_capability_decorator_self_registers():
    @reg.capability("unit_test_cap", summary="a test capability",
                    inputs=("text",), outputs=("json",), tags=("test",))
    def _f():
        return 1

    contracts = {c["name"]: c for c in reg.capability_contracts()}
    assert "unit_test_cap" in contracts
    assert contracts["unit_test_cap"]["outputs"] == ["json"]


def test_register_and_satisfiable():
    reg.register_capability(reg.CapabilityContract(
        "needs_missing", requires=("7z",)))
    reg.register_capability(reg.CapabilityContract("needs_nothing"))
    reg.refresh()
    # a contract requiring nothing is always satisfiable
    assert reg.satisfiable("needs_nothing")
    # unknown contract is not satisfiable
    assert not reg.satisfiable("does_not_exist")


def test_contract_to_dict_shape():
    c = reg.CapabilityContract("x", summary="s", inputs=("a",), outputs=("b",),
                               requires=("sandbox",), tags=("t",))
    d = c.to_dict()
    assert d == {"name": "x", "summary": "s", "inputs": ["a"],
                 "outputs": ["b"], "requires": ["sandbox"], "tags": ["t"]}
