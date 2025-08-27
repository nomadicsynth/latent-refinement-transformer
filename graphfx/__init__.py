from .core import Graph, build_model_from_graph  # re-export for convenience

__all__ = ["Graph", "build_model_from_graph"]
"""graphfx: Minimal graph -> PyTorch codegen POC.

Usage:
    from graphfx.core import Graph, build_model_from_graph

This package provides a tiny schema for nodes/edges and turns a linear-ish
graph into an nn.Module for quick prototyping. It intentionally supports a
small set of node types for simplicity.
"""

from .core import Graph, build_model_from_graph

__all__ = ["Graph", "build_model_from_graph"]
