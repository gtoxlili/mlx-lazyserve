"""Loader for Prism ML "Bonsai" ternary packs (``model_type: prism_hadamard_qwen35``).

These packs are ordinary ``mlx_lm.models.qwen3_5.TextModel`` graphs with every Linear
and Embedding swapped for a :class:`Packed` module: 2-bit affine weights (group 128)
whose activations are rotated by a fast Walsh-Hadamard transform before the quantized
matmul. The rotation is what buys the accuracy at 1.72 bits/weight, and it lives in the
*forward pass*, not in the checkpoint -- so ``mlx_lm.load`` cannot serve these weights
even if it recognized the architecture. Its model registry dispatches on ``model_type``
by importing ``mlx_lm.models.<model_type>``, and there is no ``prism_hadamard_qwen35``
module, so the load fails outright rather than silently producing garbage.

Upstream ships an equivalent loader inside the pack itself (``runtime/artifact.py``,
``runtime/runtime.py``), meant to be used via ``sys.path.insert``. We vendor it instead:
sys.path-injecting code out of a model download into a long-lived server process makes
the runtime's behaviour a property of whichever snapshot HF last handed us, and the
pack's own copy is in fact broken (see ``schema_version`` below). Vendored, the transform
is pinned, reviewable, and testable.

Deltas from the upstream runtime, both deliberate:

* ``schema_version``: upstream's ``load_model`` raises unless the value is exactly ``1``,
  but every published Bonsai 2 pack ships ``2`` -- the shipped loader rejects the weights
  it shipped with. We accept both. The v2 additions are descriptive metadata
  (``components``, ``hadamard_config``, ``vision_config``, ``tensor_namespace``); the
  ``modules`` contract this loader consumes is unchanged.
* Entry point: :func:`load` takes a repo id and returns ``(model, tokenizer)``, matching
  ``mlx_lm.load`` so :class:`~mlx_lazyserve.engine.MlxLmModel` can use it unmodified.
  Upstream's ``load_model`` takes a local directory and returns ``(model, config)``.

Served text-only. Schema v2 packs do ship a vision tower (``components.vision``), but we
drop it on load, exactly as mlx-lm's own ``qwen3_5`` sanitize does: this server pins these
models to the text engine, so the tower would be ~0.9 GiB of resident weights nothing ever
calls. Note the pack's ``PACK-RUNTIME.md`` still claims "Vision and MTP are not included";
that text is stale — trust ``config.json``'s ``components``.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

logger = logging.getLogger(__name__)

MODEL_TYPE = "prism_hadamard_qwen35"
SUPPORTED_SCHEMA_VERSIONS = (1, 2)

# Fixed by the pack format: affine 2-bit, group 128.
GROUP_SIZE = 128
BITS = 2
# Hadamard transforms are power-of-two sized; the packs only ever use these.
VALID_BLOCKS = (512, 1024, 2048, 4096)

# ``config.json``'s ``tensor_namespace`` says what the saved tensor keys are prefixed with,
# while ``modules[].path`` is always relative to TextModel itself. v1 packs saved the bare
# text namespace; v2 saves mlx-vlm's multimodal one, so the text weights sit under
# "language_model." with the vision tower beside them. Anything unrecognized is an error
# rather than a guess: silently mismatching a prefix yields a model that loads and emits noise.
TENSOR_NAMESPACE_PREFIXES = {
    None: "",
    "mlx-vlm-qwen3_5": "language_model.",
}


def fwht(x: mx.array, block: int, signs: mx.array, inverse: bool = False) -> mx.array:
    """Sign-flip and fast Walsh-Hadamard transform over the last axis, in blocks.

    Done in float32 regardless of activation dtype: the transform sums ``block`` terms,
    and at block 4096 float16 loses too much to the accumulation.
    """
    shape, dtype = x.shape, x.dtype
    if shape[-1] % block:
        raise ValueError("Hadamard block does not divide activation width")
    x = x.astype(mx.float32)
    if not inverse:
        x = x * signs
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1 / math.sqrt(block)).reshape(shape)
    if inverse:
        x = x * signs
    return x.astype(dtype)


class Packed(nn.Module):
    """A Hadamard-rotated 2-bit Linear or Embedding.

    ``block == 0`` means this module's weights were packed without a rotation, so the
    forward pass is a plain quantized matmul.
    """

    def __init__(self, arrays, block=0, signs=None, embedding=False, dtype=mx.float16):
        super().__init__()
        self.weight, self.scales, self.biases = [mx.array(a) for a in arrays]
        self.block, self.signs, self.embedding, self.dtype = block, signs, embedding, dtype

    def __call__(self, x: mx.array) -> mx.array:
        if self.embedding:
            # Embedding: gather rows, dequantize, then undo the rotation so the residual
            # stream stays in the unrotated basis the norms and RoPE expect.
            shape = x.shape
            indices = x.reshape(-1)
            out = (
                mx.dequantize(
                    self.weight[indices],
                    self.scales[indices],
                    self.biases[indices],
                    group_size=GROUP_SIZE,
                    bits=BITS,
                )
                .reshape(*shape, -1)
                .astype(self.dtype)
            )
            return fwht(out, self.block, self.signs, inverse=True) if self.block else out
        # Linear: rotate the activations into the basis the weights were packed in.
        if self.block:
            x = fwht(x, self.block, self.signs)
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=GROUP_SIZE,
            bits=BITS,
        )


def _validate_record(original, record, arrays, signs) -> None:
    """Check a pack record against the module it claims to replace.

    Every check here is a silent-corruption guard: a shape or sign-vector mismatch would
    otherwise load fine and generate noise.
    """
    if not isinstance(original, (nn.Linear, nn.Embedding)):
        raise ValueError("Unsupported packed module target")
    if record["embedding"] != isinstance(original, nn.Embedding):
        raise ValueError("Packed module kind mismatch")
    rows, width = original.weight.shape
    if width % GROUP_SIZE:
        raise ValueError("Invalid packed width")
    # 2-bit in uint32 -> 16 weights per word; one scale and one bias per group of 128.
    expected = [(rows, width // 16), (rows, width // GROUP_SIZE), (rows, width // GROUP_SIZE)]
    if [a.shape for a in arrays] != expected or arrays[0].dtype != mx.uint32:
        raise ValueError("Invalid packed tensor shapes or storage dtype")
    for array in arrays[1:]:
        if array.dtype not in (mx.float16, mx.float32, mx.bfloat16):
            raise ValueError("Invalid affine dtype")
        if not mx.all(mx.isfinite(array)).item():
            raise ValueError("Non-finite affine parameters")
    block = record["block"]
    if block:
        if width % block or signs is None or signs.shape != (width,):
            raise ValueError("Invalid transform dimensions")
        if not mx.all((signs == 1) | (signs == -1)).item():
            raise ValueError("Invalid sign values")
    elif signs is not None:
        raise ValueError("Unexpected sign vector")


def is_prism_pack(config: dict) -> bool:
    return config.get("model_type") == MODEL_TYPE


def _text_weights(weights: dict, namespace: str | None) -> dict:
    """Strip the namespace prefix off the text weights and drop everything else."""
    try:
        prefix = TENSOR_NAMESPACE_PREFIXES[namespace]
    except KeyError:
        raise ValueError(f"Unknown pack tensor namespace {namespace!r}") from None
    if not prefix:
        return weights
    # The discards are the vision tower; we serve these packs text-only.
    return {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}


def load_packed(directory: str | Path) -> tuple[TextModel, dict]:
    """Build a TextModel from a pack directory, swapping in the Packed modules."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    if not is_prism_pack(config):
        raise ValueError(f"Not a {MODEL_TYPE} pack: model_type={config.get('model_type')!r}")
    schema = config.get("schema_version")
    if schema not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported packed model schema version {schema!r}")

    model = TextModel(TextModelArgs.from_dict(config["text_config"]))
    raw = mx.load(str(directory / "model.safetensors"))
    weights = _text_weights(raw, config.get("tensor_namespace"))
    dropped = len(raw) - len(weights)

    seen: set[str] = set()
    for record in config["modules"]:
        name = record["path"]
        if name in seen:
            raise ValueError("Duplicate packed module")
        seen.add(name)
        # Walk the dotted path; numeric segments index into layer lists.
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        arrays = [weights[f"{name}.{suffix}"] for suffix in ("weight", "scales", "biases")]
        if record["dtype"] != "float16":
            raise ValueError("Unsupported activation dtype")
        block = record["block"]
        if block and block not in VALID_BLOCKS:
            raise ValueError("Unsupported block size")
        signs = weights.get(f"{name}.signs")
        if block and signs is None:
            raise ValueError("Missing sign vector")
        _validate_record(getattr(parent, parts[-1]), record, arrays, signs)
        setattr(parent, parts[-1], Packed(arrays, block, signs, record["embedding"], mx.float16))

    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    logger.info(
        "prism pack loaded: %d packed modules, schema v%s, %d non-text tensors dropped",
        len(seen),
        schema,
        dropped,
    )
    return model, config


def load(path_or_hf_repo: str):
    """``mlx_lm.load``-compatible entry point: repo id (or local dir) -> (model, tokenizer)."""
    from mlx_lm.utils import _download, load_tokenizer

    path = _download(path_or_hf_repo)
    model, _config = load_packed(path)
    return model, load_tokenizer(path)
