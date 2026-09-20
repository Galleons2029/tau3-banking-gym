"""Corpus matching, the snapshot archive, and workspace safety rails."""

import gzip
import json
from pathlib import Path

import pytest

from tau3.perturb import scan, snapshot, workspace
from tau3.perturb.materialize import rewrite_corpus
from tau3.perturb.rewrite import build_rewriter


def corpus_file(key, text):
    return scan.CorpusFile(key=key, kind=scan.kind_for(key), path=Path(key), text=text)


def test_json_escapes_do_not_hide_a_symbol():
    """Regression: raw-JSON matching silently undercounts.

    In raw JSON, ``\\n`` is a backslash followed by the letter ``n`` -- a word
    character -- so the boundary in front of the next token fails and the
    symbol is missed. Decoding first is what makes the count correct.
    """
    document = {"id": "doc_x_001", "content": "deposit holds\nRho-Bank will hold"}
    raw = json.dumps(document)
    assert "\\nRho-Bank" in raw, "the fixture must actually contain the escape"

    matchable = scan.matchable_text(corpus_file("data/documents/doc_x_001.json", raw))
    counter = build_rewriter({"Rho-Bank": "Vela-Trust"})
    assert counter.count(raw) == {}, "raw text misses it -- this is the bug"
    assert counter.count(matchable) == {"Rho-Bank": 1}


def test_rewrite_corpus_edits_json_structurally_and_text_literally():
    values = build_rewriter({"Rho-Bank": "Vela-Trust"})
    keys = build_rewriter({"rho_bank_subscription": "vela_plus_subscription"})
    files = [
        corpus_file(
            "data/documents/doc_x_001.json",
            json.dumps({"id": "doc_x_001", "content": "a\nRho-Bank b",
                        "rho_bank_subscription": True}),
        ),
        corpus_file("code/tools.py", "# Rho-Bank helper\nBRAND = 'Rho-Bank'\n"),
    ]
    outputs, hits = rewrite_corpus(files, values, keys)

    document = json.loads(outputs["data/documents/doc_x_001.json"])
    assert document["content"] == "a\nVela-Trust b"
    assert "vela_plus_subscription" in document
    assert outputs["code/tools.py"] == "# Vela-Trust helper\nBRAND = 'Vela-Trust'\n"
    assert hits["Rho-Bank"] == 3


def test_kind_for_classifies_every_corpus_location():
    assert scan.kind_for("data/documents/doc_a_001.json") == "document"
    assert scan.kind_for("data/tasks/task_001.json") == "task"
    assert scan.kind_for("data/tasks.json") == "tasks_aggregate"
    assert scan.kind_for("data/db.json") == "db"
    assert scan.kind_for("data/prompts/components/x.md") == "prompt"
    assert scan.kind_for("code/tools.py") == "python"
    assert scan.kind_for("code/GUIDE.md") == "markdown"
    with pytest.raises(ValueError):
        scan.kind_for("somewhere/else.txt")


def test_split_segment_uses_the_longest_matching_category():
    # "business_checking_accounts" must win over "checking_accounts".
    assert scan._split_segment("business_checking_accounts_navy_blue") == (
        "business_checking_accounts",
        "navy_blue",
    )
    assert scan._split_segment("checking_accounts_blue_account") == (
        "checking_accounts",
        "blue_account",
    )


def _write_archive(root: Path, files: dict[str, str], *, digest=None, hashes=None):
    import hashlib

    hashes = hashes or {
        key: hashlib.sha256(text.encode()).hexdigest() for key, text in files.items()
    }
    digest = digest or hashlib.sha256(
        "\n".join(f"{k}:{hashes[k]}" for k in sorted(hashes)).encode()
    ).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    with gzip.open(root / snapshot.ARCHIVE, "wt", encoding="utf-8") as handle:
        json.dump(
            {"schema_version": 1, "corpus_digest": digest,
             "hashes": hashes, "files": files},
            handle,
        )
    return digest


def test_snapshot_round_trips(tmp_path):
    files = {"data/db.json": '{"a": 1}', "code/tools.py": "X = 1\n"}
    digest = _write_archive(tmp_path / "snap", files)
    loaded = snapshot.load(tmp_path / "snap")
    assert loaded.digest == digest
    assert loaded.files == files
    assert loaded.file_count == 2


def test_snapshot_detects_tampered_content(tmp_path):
    files = {"data/db.json": '{"a": 1}'}
    _write_archive(tmp_path / "snap", files, hashes={"data/db.json": "0" * 64})
    with pytest.raises(ValueError, match="does not match its hash"):
        snapshot.load(tmp_path / "snap")


def test_snapshot_rejects_an_unknown_schema(tmp_path):
    root = tmp_path / "snap"
    root.mkdir()
    with gzip.open(root / snapshot.ARCHIVE, "wt", encoding="utf-8") as handle:
        json.dump({"schema_version": 99, "files": {}, "hashes": {}}, handle)
    with pytest.raises(ValueError, match="Unsupported snapshot schema"):
        snapshot.load(root)


def test_refuses_to_write_generated_files_into_the_canonical_domain():
    """environment_fingerprint() rglobs the domain; a variant nested there
    would silently change the canonical hash and invalidate every bundle."""
    data_dir = scan.domain_data_dir()
    with pytest.raises(ValueError, match="Refusing to write"):
        workspace.assert_outside_domain(data_dir / "prompts" / "variant")
    with pytest.raises(ValueError, match="Refusing to write"):
        workspace.assert_outside_domain(scan.domain_code_dir())
    # Anywhere else is fine.
    workspace.assert_outside_domain(Path("/tmp/some-variant"))
