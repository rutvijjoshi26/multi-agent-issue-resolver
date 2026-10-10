import json

from miresolver import cli
from miresolver.cli import main
from miresolver.rules import CandidateIssue, CandidatePR


def fake_corpus(count=40):
    issues, prs = [], {}
    for n in range(1, count + 1):
        issues.append(CandidateIssue(n, f"t{n}", "b", ("bug",), (1000 + n,)))
        prs[1000 + n] = CandidatePR(
            1000 + n, True, f"fix {n}", (f"src/f{n}.ts",), f"{n:040x}"
        )
    return issues, prs


def setup(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    monkeypatch.setattr(cli, "_load_corpus", lambda url: fake_corpus())


def test_benchmark_writes_a_v2_manifest_with_disjoint_sets(monkeypatch, tmp_path, capsys):
    setup(monkeypatch, tmp_path)
    out = tmp_path / "manifest.v2.json"
    code = main(["benchmark", "--out", str(out), "--seed", "11",
                 "--test-size", "10", "--dev-size", "15"])
    assert code == 0
    data = json.loads(out.read_text())
    assert data["meta"]["version"] == 2 and data["meta"]["seed"] == 11
    assert len(data["test"]) == 10 and len(data["dev"]) == 15
    assert not {c["issue_number"] for c in data["test"]} & {
        c["issue_number"] for c in data["dev"]}
    assert "test_sha256" in capsys.readouterr().out


def test_benchmark_keeps_cases_from_earlier_runs_out_of_the_test_set(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "cases.jsonl").write_text(
        "".join(json.dumps({"issue_number": n}) + "\n" for n in range(1, 26))
    )
    out = tmp_path / "m.json"
    assert main(["benchmark", "--out", str(out), "--test-size", "10",
                 "--dev-size", "15", "--exclude-run", str(run)]) == 0
    data = json.loads(out.read_text())
    assert all(c["issue_number"] > 25 for c in data["test"])
    assert data["meta"]["excluded_from_test"] == list(range(1, 26))


def test_benchmark_refuses_to_overwrite_a_frozen_manifest(monkeypatch, tmp_path, capsys):
    setup(monkeypatch, tmp_path)
    out = tmp_path / "m.json"
    out.write_text("{}")
    assert main(["benchmark", "--out", str(out)]) == 1
    assert "exists" in capsys.readouterr().err
    assert out.read_text() == "{}"
