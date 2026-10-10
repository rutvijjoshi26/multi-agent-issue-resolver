from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TARGET_REPO = "microsoft/vscode"
DEFAULT_DOTENV_PATH = Path(".env")


class ConfigError(Exception):
    """Raised when required configuration is absent."""


@dataclass(frozen=True)
class Config:
    github_token: str
    #: None when the run was told not to touch Postgres.
    database_url: str | None
    target_repo: str


def read_dotenv(path: Path | None = None) -> dict[str, str]:
    """Read KEY=VALUE lines from a .env file.

    The README tells people to copy .env.example to .env and run, so
    something has to read the file. This is a deliberate twenty lines rather
    than a python-dotenv dependency: the format the project actually uses is
    plain assignments, and the reader is easy to test offline.

    Blank lines, comments and malformed lines are ignored. Callers layer the
    real environment on top, so an exported variable always wins.
    """
    path = DEFAULT_DOTENV_PATH if path is None else path
    try:
        text = path.read_text()
    except OSError:
        return {}

    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.removeprefix("export ").strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def _required(env: Mapping[str, str], key: str) -> str:
    try:
        return env[key]
    except KeyError as exc:
        raise ConfigError(f"{key} is required") from exc


def load_config(
    env: Mapping[str, str], *, require_database_url: bool = True
) -> Config:
    return Config(
        github_token=_required(env, "GITHUB_TOKEN"),
        database_url=(
            _required(env, "DATABASE_URL") if require_database_url else None
        ),
        target_repo=env.get("TARGET_REPO", DEFAULT_TARGET_REPO),
    )
