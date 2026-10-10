import contextlib
import json
import re

import pytest
from agent_fakes import FakeReader, ScriptedChat, hit

from miresolver import cli, runtime
from miresolver.agents.graph import ResolverDeps
from miresolver.cli import main

MODELS_YAML = """
tiers:
  embed: {provider: openai-compatible, base_url: "http://x/v1", model: emb, api_key_env: K,
          input_cost_per_mtok: 0.02, output_cost_per_mtok: 0}
  screen: {provider: openai-compatible, base_url: "http://x/v1", model: small, api_key_env: K,
           input_cost_per_mtok: 0.15, output_cost_per_mtok: 0.6}
  investigate: {provider: openai-compatible, base_url: "http://x/v1", model: small, api_key_env: K,
                input_cost_per_mtok: 0.15, output_cost_per_mtok: 0.6}
  synthesize: {provider: openai-compatible, base_url: "http://x/v1", model: big, api_key_env: K,
               input_cost_per_mtok: 2.5, output_cost_per_mtok: 10}
"""
SHA = "d" * 40


@pytest.fixture
def models(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(MODELS_YAML)
    return path


def screen_all(prompt):
    return {"scores": [{"issue": int(n), "score": 0.9}
                       for n in re.findall(r"^#(\d+):", prompt, re.M)]}


def verdict(conf):
    def respond(prompt):
        n = int(re.search(r"PAST ISSUE #(\d+)", prompt).group(1))
        return {"applies": True, "confidence": conf, "files": [f"src/f{n}.ts"],
                "approach": "guard it", "reasoning": "r"}
    return respond


def install_fake_runtime(monkeypatch, *, conf=0.9, synth_conf=0.8, hits=None):
    built = {}

    @contextlib.asynccontextmanager
    async def fake_open_runtime(settings, env, **kwargs):
        built["kwargs"] = kwargs
        async def retrieve(query, limit):
            return (hits if hits is not None else [hit(1, pr=10), hit(2, pr=20)])[:limit]
        yield ResolverDeps(
            retriever=retrieve,
            screen_chat=ScriptedChat(respond=screen_all),
            investigate_chat=ScriptedChat(respond=verdict(conf)),
            synthesize_chat=ScriptedChat(respond=lambda p: {
                "approach": "guard the null", "files": ["src/f1.ts"],
                "cited_issues": [1], "confidence": synth_conf}),
            reader=FakeReader(),
            options=kwargs["options"],
        )

    monkeypatch.setattr(runtime, "open_runtime", fake_open_runtime)
    return built


def test_resolve_prints_files_approach_and_citations(models, monkeypatch, capsys):
    built = install_fake_runtime(monkeypatch)
    code = main(["resolve", "Crash on save", "--body", "trace", "--base-commit", SHA,
                 "--models", str(models)])
    out = capsys.readouterr().out
    assert code == 0
    assert "src/f1.ts" in out and "guard the null" in out
    assert "#1" in out and "PR 10" in out
    assert built["kwargs"]["repo"] == "microsoft/vscode"


def test_resolve_json_output(models, monkeypatch, capsys):
    install_fake_runtime(monkeypatch)
    main(["resolve", "Crash", "--base-commit", SHA, "--models", str(models), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["files"] == ["src/f1.ts"] and data["abstained"] is False
    assert data["citations"] == [{"issue_number": 1, "pr_number": 10}]


def test_resolve_reports_an_abstention_without_recommending_files(models, monkeypatch, capsys):
    install_fake_runtime(monkeypatch, conf=0.2)
    code = main(["resolve", "Crash", "--base-commit", SHA, "--models", str(models)])
    out = capsys.readouterr().out
    assert code == 0
    assert "No confident match" in out
    assert "Suggested files" not in out


def test_resolve_requires_a_base_commit(models, capsys):
    with pytest.raises(SystemExit):
        main(["resolve", "Crash", "--models", str(models)])


def test_resolve_needs_a_title_or_an_issue_number(models, capsys):
    assert main(["resolve", "--base-commit", SHA, "--models", str(models)]) == 1
    assert "title" in capsys.readouterr().err


def test_resolve_reports_a_bad_models_file_cleanly(tmp_path, capsys):
    code = main(["resolve", "x", "--base-commit", SHA, "--models", str(tmp_path / "no.yaml")])
    assert code == 1 and "no.yaml" in capsys.readouterr().err


def write_manifest(tmp_path, n=6):
    cases = [{"issue_number": i, "pr_number": 100 + i, "base_commit": SHA,
              "changed_files": [f"src/f{i}.ts"]} for i in range(1, n + 1)]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"dev": cases[:4], "full": cases,
                                "meta": {"generated_at": "g", "seed": 5}}))
    return path


def test_evaluate_estimate_prints_cost_without_building_clients(models, tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("no clients for an estimate")
    monkeypatch.setattr(runtime, "open_runtime", boom)
    code = main(["evaluate", "--estimate", "--models", str(models),
                 "--manifest", str(write_manifest(tmp_path)), "--limit", "2"])
    data = json.loads(capsys.readouterr().out)
    assert code == 0 and data["cases"] == 2 and data["priced_dollars"] > 0


def test_evaluate_refuses_the_full_set_without_allow_full(models, tmp_path, capsys):
    code = main(["evaluate", "--set", "full", "--estimate", "--models", str(models),
                 "--manifest", str(write_manifest(tmp_path))])
    assert code == 1 and "allow-full" in capsys.readouterr().err


def test_evaluate_needs_database_url(models, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    code = main(["evaluate", "--models", str(models),
                 "--manifest", str(write_manifest(tmp_path))])
    assert code == 1 and "DATABASE_URL" in capsys.readouterr().err


def test_evaluate_runs_and_writes_results(models, tmp_path, monkeypatch, capsys):
    install_fake_runtime(monkeypatch, hits=[hit(1, pr=10, files=("src/f3.ts",)), hit(2, pr=20)])
    monkeypatch.setattr(cli, "_load_case_texts",
                        lambda url, numbers: {n: (f"t{n}", "b") for n in numbers})
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    out_root = tmp_path / "results"
    code = main(["evaluate", "--models", str(models), "--manifest", str(write_manifest(tmp_path)),
                 "--out-root", str(out_root), "--seed", "9", "--limit", "3"])
    assert code == 0
    [run_dir] = list(out_root.iterdir())
    assert run_dir.name.startswith("dev-n3-seed9-")
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["counts"]["scored"] == 3
    assert summary["run"]["manifest_generated_at"] == "g"
    assert summary["run"]["options"]["retrieve_k"] == 10
    assert summary["models"]["synthesize"]["model"] == "big"
    assert str(run_dir) in capsys.readouterr().out
