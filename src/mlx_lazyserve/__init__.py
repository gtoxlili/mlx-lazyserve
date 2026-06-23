"""mlx-lazyserve: an Ollama-style, lazy-loading MLX inference server.

Loads models on the first request, serves an OpenAI-compatible API, and unloads
them after a configurable idle timeout to free unified memory.
"""

from .__main__ import main

__all__ = ["main"]
__version__ = "0.1.0"
