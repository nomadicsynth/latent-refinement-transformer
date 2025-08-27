from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Node:
    id: str
    type: str
    params: Dict[str, float | int | str] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    label: Optional[str] = None


@dataclass
class Graph:
    nodes: List[Node]
    edges: List[Edge]

    @staticmethod
    def from_json(data: Dict) -> "Graph":
        nodes = [Node(n["id"], n["type"], n.get("data", {})) for n in data.get("nodes", [])]
        edges = [Edge(e["source"], e["target"], e.get("label")) for e in data.get("edges", [])]
        return Graph(nodes=nodes, edges=edges)


class Embed(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=(attn_mask == 0) if attn_mask is not None else None,
            need_weights=False,
        )
        x = x + attn_out
        h = self.ln2(x)
        x = x + self.mlp(h)
        return x


class Norm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


class LMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


REGISTRY = {
    "Embedding": Embed,
    "Block": TransformerBlock,
    "Norm": Norm,
    "LMHead": LMHead,
}


class GraphModule(nn.Module):
    def __init__(self, modules: Dict[str, nn.Module], order: List[Tuple[str, Optional[str]]]):
        super().__init__()
        for name, mod in modules.items():
            self.add_module(name, mod)
        self._order = order

    def forward(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cache: Dict[str, torch.Tensor] = dict(tensors)
        for nid, mask_id in self._order:
            mod: nn.Module = getattr(self, nid)
            if isinstance(mod, Embed):
                x = cache.get("input_ids")
                cache[nid] = mod(x)
            elif isinstance(mod, TransformerBlock):
                x = cache.get(self._prev_of(nid))
                mask = cache.get(mask_id) if mask_id else None
                cache[nid] = mod(x, mask)
            elif isinstance(mod, Norm):
                x = cache.get(self._prev_of(nid))
                cache[nid] = mod(x)
            elif isinstance(mod, LMHead):
                x = cache.get(self._prev_of(nid))
                cache[nid] = mod(x)
        return cache

    def _prev_of(self, nid: str) -> str:
        idx = [i for i, (id_, _) in enumerate(self._order) if id_ == nid][0]
        if idx == 0:
            return "input_embeds"
        return self._order[idx - 1][0]


def topological_order(g: Graph) -> List[str]:
    indeg: Dict[str, int] = {n.id: 0 for n in g.nodes}
    for e in g.edges:
        indeg[e.target] = indeg.get(e.target, 0) + 1
    queue = [nid for nid, d in indeg.items() if d == 0]
    out: List[str] = []
    while queue:
        cur = queue.pop(0)
        out.append(cur)
        for e in g.edges:
            if e.source == cur:
                indeg[e.target] -= 1
                if indeg[e.target] == 0:
                    queue.append(e.target)
    known = set(out)
    for n in g.nodes:
        if n.id not in known:
            out.append(n.id)
    return out


def build_model_from_graph(g: Graph) -> GraphModule:
    modules: Dict[str, nn.Module] = {}
    params_for = {n.id: n.params for n in g.nodes}
    vocab_size = int(params_for.get("Embedding", {}).get("vocab_size", 256))
    hidden = int(params_for.get("Block", {}).get("hidden_size", 256))
    heads = int(params_for.get("Block", {}).get("num_heads", 8))

    for n in g.nodes:
        p = n.params
        if n.type == "Embedding":
            modules[n.id] = Embed(int(p.get("vocab_size", vocab_size)), int(p.get("hidden_size", hidden)))
        elif n.type == "Block":
            modules[n.id] = TransformerBlock(int(p.get("hidden_size", hidden)), int(p.get("num_heads", heads)))
        elif n.type == "Norm":
            modules[n.id] = Norm(int(p.get("hidden_size", hidden)))
        elif n.type == "LMHead":
            modules[n.id] = LMHead(int(p.get("hidden_size", hidden)), int(p.get("vocab_size", vocab_size)))

    order_ids = topological_order(g)
    mask_map: Dict[str, Optional[str]] = {nid: None for nid in order_ids}
    for e in g.edges:
        if e.label and e.label.lower() == "mask":
            mask_map[e.target] = e.source

    order: List[Tuple[str, Optional[str]]] = [(nid, mask_map.get(nid)) for nid in order_ids if nid in modules]
    return GraphModule(modules, order)


def run_demo(json_path: str, batch_size: int = 1, seq_len: int = 16, device: str = "cpu"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    g = Graph.from_json(data)
    model = build_model_from_graph(g)
    model.to(device)
    vocab = int(data.get("defaults", {}).get("vocab_size", 256))
    x = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    attn = torch.ones(batch_size, seq_len, device=device)
    cache = {"input_ids": x, "attention_mask": attn}
    out = model(cache)
    last = None
    for k in out:
        last = k
    y = out.get(last)
    print({"last": last, "shape": tuple(y.shape) if isinstance(y, torch.Tensor) else None})
