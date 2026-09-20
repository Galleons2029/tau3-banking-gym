"""Verification stages that do not need a real variant build."""

import json

from tau3.perturb import verify
from tau3.perturb.models import ConceptMap, FormMap, PerturbationPlan


def make_plan(variant_id="aaaa1111", pairs=(("Rho-Bank", "Vela-Trust"),)):
    return PerturbationPlan(
        variant_id=variant_id,
        domain_name=f"banking_knowledge__{variant_id}",
        package_name=f"tau3_bk_{variant_id}",
        corpus_digest="0" * 64,
        seed=1,
        recipe={"name": "names_only"},
        concepts=[
            ConceptMap(
                concept_id=f"c{index}",
                kind="brand",
                root=source,
                new_root=target,
                forms=[FormMap(source=source, target=target, style="literal", count=1)],
            )
            for index, (source, target) in enumerate(pairs)
        ],
    )


def build_variant(root, files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "plan.json").write_text("{}")


def test_leak_scan_passes_on_a_fully_renamed_variant(tmp_path):
    build_variant(tmp_path, {
        "data/documents/doc_a_001.json": json.dumps(
            {"id": "doc_a_001", "content": "Vela-Trust will hold the deposit"}
        ),
        "code/tools.py": "BRAND = 'Vela-Trust'\n",
    })
    assert verify.check_leaks(tmp_path, make_plan()) == []


def test_leak_scan_catches_a_missed_rename_in_content(tmp_path):
    build_variant(tmp_path, {
        "data/documents/doc_a_001.json": json.dumps(
            {"id": "doc_a_001", "content": "Rho-Bank will hold the deposit"}
        ),
    })
    problems = verify.check_leaks(tmp_path, make_plan())
    assert len(problems) == 1
    assert "canonical symbol 'Rho-Bank'" in problems[0]


def test_leak_scan_sees_through_json_escapes(tmp_path):
    """A symbol hidden behind an escape must not slip past the scan."""
    build_variant(tmp_path, {
        "data/documents/doc_a_001.json": json.dumps(
            {"id": "doc_a_001", "content": "holds\nRho-Bank will"}
        ),
    })
    problems = verify.check_leaks(tmp_path, make_plan())
    assert problems and "Rho-Bank" in problems[0]


def test_leak_scan_checks_filenames_too(tmp_path):
    # A document id lives in its filename as well as its body.
    plan = make_plan(pairs=(("doc_checking_accounts_light_green_account_001",
                             "doc_checking_accounts_light_alder_account_001"),))
    build_variant(tmp_path, {
        "data/documents/doc_checking_accounts_light_green_account_001.json":
            json.dumps({"id": "doc_checking_accounts_light_alder_account_001"}),
    })
    problems = verify.check_leaks(tmp_path, plan)
    assert any("in filename" in problem for problem in problems)


def test_leak_scan_ignores_bytecode_from_running_the_variant(tmp_path):
    build_variant(tmp_path, {"code/tools.py": "BRAND = 'Vela-Trust'\n"})
    cache = tmp_path / "code" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "tools.cpython-312.pyc").write_bytes(b"\xcb\x00binary Rho-Bank")
    assert verify.check_leaks(tmp_path, make_plan()) == []


def test_disjoint_flags_a_shared_symbol_between_variants():
    train = make_plan("aaaa1111", pairs=(("Rho-Bank", "Vela-Trust"),))
    evaluation = make_plan("bbbb2222", pairs=(("Rho-Bank", "Vela-Trust"),))
    problems = verify.check_disjoint(train, [evaluation])
    assert problems and "shares 1 symbol" in problems[0]


def test_disjoint_passes_for_genuinely_separate_variants():
    train = make_plan("aaaa1111", pairs=(("Rho-Bank", "Vela-Trust"),))
    evaluation = make_plan("bbbb2222", pairs=(("Rho-Bank", "Kestrel-Union"),))
    assert verify.check_disjoint(train, [evaluation]) == []


def test_disjoint_ignores_the_variant_itself():
    train = make_plan("aaaa1111")
    assert verify.check_disjoint(train, [train]) == []


def test_report_is_ok_only_when_every_stage_is_clean():
    report = verify.Report(variant_id="aaaa1111")
    report.record("leak", [])
    assert report.ok
    report.record("structure", ["something broke"])
    assert not report.ok
    assert report.to_dict()["stages"]["structure"] == "1 finding(s)"
