import json

from fademem.cli import main


def test_show_config(in_repo_root, capsys):
    assert main(["show-config", "--profile", "cloud-openai"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["profile"]["memory_llm"]["model"] == "gpt-4o-mini"
    assert out["params"]["layers"]["theta_promote"] == 0.7


def test_check_providers_with_fakes(in_repo_root, capsys):
    assert main(["check-providers", "--profile", "test"]) == 0
    assert "FAIL" not in capsys.readouterr().out
