"""Config loading: YAML -> attribute-access object, with dotted CLI overrides."""
from __future__ import annotations
from typing import Any, Dict, Iterable, Tuple
import yaml


class Cfg:
    def __init__(self, d: Dict[str, Any]):
        object.__setattr__(self, "_d", {})
        for k, v in d.items():
            self._d[k] = Cfg(v) if isinstance(v, dict) else v

    def __getattr__(self, k):
        try:
            return self._d[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self._d[k] = v

    def __contains__(self, k):
        return k in self._d

    def get(self, k, default=None):
        return self._d.get(k, default)

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v.to_dict() if isinstance(v, Cfg) else v) for k, v in self._d.items()}

    def __repr__(self):
        return f"Cfg({self.to_dict()})"


def parse_override(s: str) -> Tuple[str, Any]:
    if "=" not in s:
        raise ValueError(f"override must look like a.b=value, got {s!r}")
    k, v = s.split("=", 1)
    return k.strip(), yaml.safe_load(v)


def set_dotted(d: Dict[str, Any], key: str, val: Any) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            raise KeyError(f"unknown config section {p!r} in override {key!r}")
        cur = cur[p]
    if parts[-1] not in cur:
        raise KeyError(f"unknown config key {key!r} (typo?)")
    cur[parts[-1]] = val


def load_config(path: str, overrides: Iterable[str] = ()) -> Cfg:
    with open(path) as f:
        d = yaml.safe_load(f)
    for o in overrides:
        k, v = parse_override(o)
        set_dotted(d, k, v)
    return Cfg(d)
