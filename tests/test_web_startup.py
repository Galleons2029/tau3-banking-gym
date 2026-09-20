"""Regression tests for the lightweight web-console startup path."""

import json
import subprocess
import sys
import threading

from fastapi.testclient import TestClient

from tau3.data_model.persona import PersonaConfig, Verbosity
from tau3.user.user_simulator import UserSimulator
from tau3.user_prompt import build_user_system_prompt
from tau3.web_console.app import create_app
from tau3.web_console.store import ArtifactStore


def test_web_import_path_skips_llm_and_dataframe_stacks():
    script = r"""
import json
import sys

import tau3
from tau3 import Task
import tau3.cli as cli

captured = {}
cli.run_web_console = lambda args: captured.update(vars(args))
sys.argv = ["tau3", "web", "--host", "0.0.0.0", "--port", "8123"]
cli.main()
import tau3.web_console.app

payload = {
    "captured": captured,
    "task": Task.__name__,
    "loaded": {
        name: name in sys.modules
        for name in (
            "litellm",
            "openai",
            "pandas",
            "deepdiff",
            "tau3.synthesis.cli",
            "tau3.worldgen.cli",
        )
    },
}
from tau3 import TextRunConfig
payload["config"] = TextRunConfig.__name__
print(json.dumps(payload))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    assert payload["captured"]["host"] == "0.0.0.0"
    assert payload["captured"]["port"] == 8123
    assert payload["task"] == "Task"
    assert payload["config"] == "TextRunConfig"
    assert not any(payload["loaded"].values())


def test_lightweight_prompt_builder_matches_user_simulator():
    persona = PersonaConfig(verbosity=Verbosity.MINIMAL)
    instructions = "Persona:\n\tCareful customer\nInstructions:\n\tCheck a transfer"

    for tools in (None, []):
        simulator = UserSimulator(
            llm="test-model",
            instructions=instructions,
            tools=tools,
            persona_config=persona,
        )
        assert simulator.system_prompt == build_user_system_prompt(
            instructions,
            persona_guidelines=persona.to_guidelines_text(),
            use_tools=tools is not None,
        )


def test_health_is_available_while_background_index_is_running(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_refresh(self, force=False):
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr(ArtifactStore, "_refresh_now", slow_refresh)
    app = create_app(tmp_path, refresh_seconds=5)

    try:
        assert started.wait(timeout=1)
        with TestClient(app) as client:
            response = client.get("/api/web/v1/health")
        assert response.status_code == 200
        assert response.json()["index"]["status"] == "indexing"
    finally:
        release.set()
