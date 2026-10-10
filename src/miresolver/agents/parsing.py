from pathlib import PurePosixPath


def clamp_unit(value: object) -> float:
    """A 0..1 score from model output. Raises ValueError if not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"not a number: {value!r}")
    return min(1.0, max(0.0, float(value)))


def clean_paths(values: object) -> tuple[str, ...]:
    """Repository-relative paths from model output, deduplicated in order.

    Anything that is not a plausible relative path is dropped rather than
    passed on, since these strings are scored and shown to users.
    """
    if not isinstance(values, list):
        return ()
    seen: dict[str, None] = {}
    for value in values:
        if not isinstance(value, str):
            continue
        path = value.strip().removeprefix("./")
        if not path:
            continue
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts:
            continue
        seen.setdefault(path, None)
    return tuple(seen)
