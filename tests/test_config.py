import pytest
from miresolver.config import Config, ConfigError, load_config, read_dotenv


def test_load_config_reads_required_values():
    cfg = load_config({
        "GITHUB_TOKEN": "ghp_example",
        "DATABASE_URL": "postgresql://localhost/test",
        "TARGET_REPO": "microsoft/vscode",
    })
    assert cfg == Config(
        github_token="ghp_example",
        database_url="postgresql://localhost/test",
        target_repo="microsoft/vscode",
    )


def test_load_config_defaults_target_repo():
    cfg = load_config({
        "GITHUB_TOKEN": "ghp_example",
        "DATABASE_URL": "postgresql://localhost/test",
    })
    assert cfg.target_repo == "microsoft/vscode"


def test_load_config_raises_on_missing_token():
    with pytest.raises(ConfigError, match="GITHUB_TOKEN"):
        load_config({"DATABASE_URL": "postgresql://localhost/test"})


def test_database_url_is_optional_when_not_required():
    cfg = load_config({"GITHUB_TOKEN": "ghp_example"}, require_database_url=False)
    assert cfg.database_url is None


def test_database_url_is_required_by_default():
    with pytest.raises(ConfigError, match="DATABASE_URL"):
        load_config({"GITHUB_TOKEN": "ghp_example"})


def test_read_dotenv_parses_assignments(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# a comment\n"
        "\n"
        "GITHUB_TOKEN=ghp_example\n"
        "export DATABASE_URL='postgresql://localhost/test'\n"
        'TARGET_REPO="microsoft/vscode"\n'
        "MALFORMED\n"
    )
    assert read_dotenv(path) == {
        "GITHUB_TOKEN": "ghp_example",
        "DATABASE_URL": "postgresql://localhost/test",
        "TARGET_REPO": "microsoft/vscode",
    }


def test_read_dotenv_returns_empty_when_file_is_absent(tmp_path):
    assert read_dotenv(tmp_path / "nope.env") == {}


def test_read_dotenv_does_not_override_the_real_environment():
    env = {"GITHUB_TOKEN": "from-shell"}
    merged = {**read_dotenv(None), **env}
    assert merged["GITHUB_TOKEN"] == "from-shell"
