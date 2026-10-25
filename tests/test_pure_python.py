import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, pathlib
from ddkg.config import load_config
from ddkg.data.vocab import Vocabulary
from ddkg.data.karpathy import load_karpathy
import pytest


def test_config_overrides(tmp_path):
    BASE = pathlib.Path(__file__).resolve().parents[1] / 'configs' / 'base.yaml'
    cfg = load_config(str(BASE), ["model.hidden_dim=128", "knowledge_graph.static_gate=true"])
    assert cfg.model.hidden_dim == 128 and cfg.knowledge_graph.static_gate is True
    with pytest.raises(KeyError):
        load_config(str(BASE), ["model.hidden_dimm=1"])


def test_vocab_roundtrip(tmp_path):
    v = Vocabulary.build([["a", "dog", "runs"], ["a", "dog", "sits"]], min_freq=2)
    ids = v.encode(["a", "dog", "flies"], 8)
    assert len(ids) == 8 and ids[0] == 1 and v.decode(ids) == "a dog <unk>"
    v.save(tmp_path / "v.json")
    assert Vocabulary.load(tmp_path / "v.json").itos == v.itos


def test_karpathy(tmp_path):
    d = {"images": [{"split": "restval", "filepath": "val2014", "filename": "x.jpg", "cocoid": 7,
                     "sentences": [{"tokens": ["a", "cat"], "raw": "A cat."}]}]}
    p = tmp_path / "k.json"; p.write_text(json.dumps(d))
    r = load_karpathy(str(p), "coco", "/root")
    assert r["train"][0]["id"] == 7 and r["train"][0]["path"] == "/root/val2014/x.jpg"
