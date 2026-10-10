from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from miresolver.models.chat import Usage

TIER_NAMES: tuple[str, ...] = ("embed", "screen", "investigate", "synthesize")


class ModelConfigError(Exception):
    """Raised when model configuration is absent or malformed."""


@dataclass(frozen=True)
class TierSettings:
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    #: US dollars per million tokens. Optional: used only to report and cap
    #: spend, and a tier without prices reports no dollar figure at all.
    input_cost_per_mtok: float | None = None
    output_cost_per_mtok: float | None = None

    def cost(self, usage: "Usage") -> float | None:
        if self.input_cost_per_mtok is None or self.output_cost_per_mtok is None:
            return None
        return (
            usage.input_tokens * self.input_cost_per_mtok
            + usage.output_tokens * self.output_cost_per_mtok
        ) / 1_000_000


@dataclass(frozen=True)
class ModelSettings:
    tiers: Mapping[str, TierSettings]

    def tier(self, name: str) -> TierSettings:
        try:
            return self.tiers[name]
        except KeyError as exc:
            raise ModelConfigError(f"unknown tier: {name}") from exc


def _tier_from_mapping(name: str, raw: object) -> TierSettings:
    if not isinstance(raw, Mapping):
        raise ModelConfigError(f"tier {name} must be a mapping")
    for required in ("provider", "model"):
        if not raw.get(required):
            raise ModelConfigError(f"tier {name} is missing {required}")
    return TierSettings(
        provider=str(raw["provider"]),
        model=str(raw["model"]),
        base_url=raw.get("base_url"),
        api_key_env=raw.get("api_key_env"),
        input_cost_per_mtok=_price(name, raw, "input_cost_per_mtok"),
        output_cost_per_mtok=_price(name, raw, "output_cost_per_mtok"),
    )


def _price(name: str, raw: Mapping, field: str) -> float | None:
    value = raw.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelConfigError(f"tier {name} {field} must be a number")
    return float(value)


def load_model_settings(text: str) -> ModelSettings:
    document = yaml.safe_load(text) or {}
    if not isinstance(document, Mapping):
        raise ModelConfigError("config must be a YAML mapping at the top level")
    raw_tiers = document.get("tiers")
    if not isinstance(raw_tiers, Mapping) or not raw_tiers:
        raise ModelConfigError("config must define a non-empty tiers mapping")

    tiers = {
        name: _tier_from_mapping(name, raw)
        for name, raw in raw_tiers.items()
    }
    missing = [name for name in TIER_NAMES if name not in tiers]
    if missing:
        raise ModelConfigError(f"config is missing tiers: {', '.join(missing)}")
    return ModelSettings(tiers=MappingProxyType(tiers))
