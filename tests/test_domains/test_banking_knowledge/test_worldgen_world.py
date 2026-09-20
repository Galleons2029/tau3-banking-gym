"""World bundle integrity and the synth domain that serves it."""

import json

import pytest

from tau3.domains.banking_knowledge.data_model import TransactionalDB
from tau3.worldgen.workflow import read_config, run_init_world, run_publish
from tau3.worldgen.world import (
    DRAFT_ENV_VAR,
    WORLD_ENV_VAR,
    load_world,
    read_world_manifest,
    write_world_manifest,
)

USER = {
    "name": "Dana Whitfield",
    "user_id": "dw41c8e2a7",
    "address": "88 Copper Lane, Boise, ID 83702",
    "email": "dana.whitfield@example.net",
    "phone_number": "208-555-0132",
    "date_of_birth": "07/22/1989",
}


def build_world(root, salt="toy-salt"):
    """A three-document world: enough to load, retrieve and replay against."""
    config = read_config()
    config = config.model_copy(update={"suffix_salt": salt})
    run_init_world(root, "toy", config)
    manifest = read_world_manifest(root)
    alias = next(
        a for a, o in manifest.alias_map.items() if o == "close_bank_account_7392"
    )

    documents = [
        (
            "doc_toy_accounts_copper_account_001",
            "Copper Account at a Glance",
            "## Fees and Limits\n\n| Item | Value |\n|---|---|\n| Monthly maintenance fee | $0 |\n",
        ),
        (
            "doc_toy_accounts_copper_account_002",
            "Internal: Closing a Copper Account",
            f"## Closure Procedure\n\n1. Verify the customer.\n2. Use {alias} to close the account.\n",
        ),
        (
            "doc_toy_accounts_copper_account_003",
            "How do I close my Copper Account?",
            "Contact support and confirm the closure reason.\n",
        ),
    ]
    for doc_id, title, content in documents:
        (root / "documents" / f"{doc_id}.json").write_text(
            json.dumps({"id": doc_id, "title": title, "content": content})
        )

    db = TransactionalDB()
    db.users.data[USER["user_id"]] = dict(USER)
    db.accounts.data["01"] = {
        "account_id": "01",
        "user_id": USER["user_id"],
        "class": "checking",
        "level": "Copper Account",
        "date_opened": "09/02/2025",
        "status": "OPEN",
        "current_holdings": "0",
    }
    (root / "db.json").write_text(json.dumps(db.model_dump(mode="json")))

    (root / "tasks" / "task_001.json").write_text(
        json.dumps(
            {
                "id": "task_001",
                "description": {"purpose": "Close a Copper Account"},
                "user_scenario": {
                    "instructions": "You want to close your Copper Account.",
                },
                "evaluation_criteria": {"actions": [], "reward_basis": ["DB"]},
                "required_documents": [documents[1][0]],
            }
        )
    )
    return root, alias


@pytest.fixture
def published(tmp_path, monkeypatch):
    # This legacy fixture exercises development loading and artifact integrity,
    # not certification. Production publication is covered by V2 tests.
    root, alias = build_world(tmp_path / "toy")
    from tau3.worldgen.world import artifact_hashes

    manifest = read_world_manifest(root)
    manifest.artifact_hashes = artifact_hashes(root)
    write_world_manifest(root, manifest)
    monkeypatch.setenv(DRAFT_ENV_VAR, "1")
    return root, alias


def test_legacy_development_integrity_is_not_publication(published):
    root, _ = published
    manifest = read_world_manifest(root)
    assert manifest.status == "draft"
    with pytest.raises(ValueError, match="certificates"):
        run_publish(root)
    assert set(manifest.artifact_hashes) == {"documents", "db.json", "tasks"}
    assert load_world(root).root == root


def test_draft_worlds_are_rejected_unless_explicitly_allowed(tmp_path, monkeypatch):
    root, _ = build_world(tmp_path / "draft")
    with pytest.raises(ValueError, match="publication gates"):
        load_world(root)
    monkeypatch.setenv(DRAFT_ENV_VAR, "1")
    assert load_world(root).manifest.status == "draft"


def test_publishing_an_empty_world_fails(tmp_path):
    config = read_config()
    root = tmp_path / "empty"
    run_init_world(root, "empty", config)
    with pytest.raises(ValueError, match="no documents"):
        run_publish(root)


def test_altered_artifacts_are_rejected(published):
    root, _ = published
    extra = root / "documents" / "doc_toy_accounts_copper_account_004.json"
    extra.write_text(json.dumps({"id": "x", "title": "x", "content": "x"}))
    with pytest.raises(ValueError, match="artifacts changed"):
        load_world(root)


def test_alias_map_colliding_with_official_names_is_rejected(published):
    root, _ = published
    manifest = read_world_manifest(root)
    manifest.alias_map["close_bank_account_7392"] = "close_bank_account_7392"
    write_world_manifest(root, manifest)
    with pytest.raises(ValueError, match="collides with official"):
        load_world(root)


def test_init_world_refuses_to_overwrite(tmp_path):
    root, _ = build_world(tmp_path / "toy")
    with pytest.raises(ValueError, match="already exists"):
        run_init_world(root, "toy", read_config())


def test_synth_domain_serves_the_world_not_the_official_corpus(published, monkeypatch):
    root, alias = published
    monkeypatch.setenv(WORLD_ENV_VAR, str(root))
    from tau3.domains.banking_synth.environment import get_environment, get_tasks

    tasks = get_tasks()
    assert [task.id for task in tasks] == ["task_001"]

    env = get_environment(task=tasks[0])
    assert env.domain_name == "banking_synth"
    # The world's own tools resolve; the official suffix does not.
    assert env.tools.has_discoverable_tool(alias)
    assert not env.tools.has_discoverable_tool("close_bank_account_7392")
    # The world's own database is in play, not the official one.
    assert set(env.tools.db.users.data) == {USER["user_id"]}


def test_synth_domain_requires_a_configured_world(monkeypatch):
    monkeypatch.delenv(WORLD_ENV_VAR, raising=False)
    from tau3.domains.banking_synth.environment import get_world

    with pytest.raises(ValueError, match=WORLD_ENV_VAR):
        get_world()
