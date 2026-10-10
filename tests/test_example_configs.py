import json
from pathlib import Path

from miresolver.agents.types import ResolverOptions
from miresolver.evaluation.runner import estimate_cost
from miresolver.models.settings import load_model_settings

ROOT = Path(__file__).parent.parent


def test_both_example_configs_load():
    for name in ("models.example.yaml", "models.openai.example.yaml"):
        load_model_settings((ROOT / "config" / name).read_text())


def test_openai_example_dev_set_upper_bound_stays_inside_the_budget():
    settings = load_model_settings((ROOT / "config/models.openai.example.yaml").read_text())
    dev = len(json.loads((ROOT / "eval/manifest.json").read_text())["dev"])
    estimate = estimate_cost(dev, settings, ResolverOptions())
    assert estimate["unpriced_tiers"] == []
    assert estimate["priced_dollars"] < 10
