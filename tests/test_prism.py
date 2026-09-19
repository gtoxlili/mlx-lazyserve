import json
import tempfile
import unittest

from pathlib import Path
from unittest import mock

import mlx.core as mx

from mlx import nn

from mlx_lazyserve import prism


class FwhtTests(unittest.TestCase):
    """The transform is the whole correctness story: get it wrong and the model still
    loads, still generates, and quietly emits noise."""

    def test_inverse_recovers_input(self):
        block = 512
        mx.random.seed(0)
        x = mx.random.normal((3, block * 2)).astype(mx.float32)
        signs = mx.where(mx.random.uniform(shape=(block * 2,)) > 0.5, 1.0, -1.0)
        back = prism.fwht(prism.fwht(x, block, signs), block, signs, inverse=True)
        self.assertLess(mx.max(mx.abs(back - x)).item(), 1e-4)

    def test_rejects_width_not_divisible_by_block(self):
        with self.assertRaises(ValueError):
            prism.fwht(mx.zeros((1, 513)), 512, mx.ones((513,)))

    def test_runs_in_float32_regardless_of_input_dtype(self):
        # float16 accumulation over 4096 terms loses too much; the transform upcasts
        # internally but must hand back the caller's dtype.
        out = prism.fwht(mx.zeros((1, 512), dtype=mx.float16), 512, mx.ones((512,)))
        self.assertEqual(out.dtype, mx.float16)


class TextWeightsTests(unittest.TestCase):
    """Schema v2 packs save tensors in mlx-vlm's namespace while module records stay
    relative to TextModel, so the prefix has to be stripped or nothing matches."""

    weights = {
        "language_model.lm_head.weight": 1,
        "language_model.model.layers.0.mlp.up_proj.weight": 2,
        "vision_tower.patch_embed.proj.weight": 3,
    }

    def test_v2_namespace_strips_prefix_and_drops_vision_tower(self):
        out = prism._text_weights(self.weights, "mlx-vlm-qwen3_5")
        self.assertEqual(
            set(out), {"lm_head.weight", "model.layers.0.mlp.up_proj.weight"}
        )

    def test_absent_namespace_is_passed_through_untouched(self):
        self.assertEqual(prism._text_weights(self.weights, None), self.weights)

    def test_unknown_namespace_raises_rather_than_guessing(self):
        with self.assertRaises(ValueError):
            prism._text_weights(self.weights, "some-future-namespace")


class ValidateRecordTests(unittest.TestCase):
    ROWS, WIDTH, BLOCK = 8, 512, 512

    def _arrays(self):
        return [
            mx.zeros((self.ROWS, self.WIDTH // 16), dtype=mx.uint32),
            mx.zeros((self.ROWS, self.WIDTH // 128), dtype=mx.float16),
            mx.zeros((self.ROWS, self.WIDTH // 128), dtype=mx.float16),
        ]

    def _record(self, **over):
        record = {"path": "x", "block": self.BLOCK, "embedding": False, "dtype": "float16"}
        record.update(over)
        return record

    def test_accepts_a_well_formed_linear_record(self):
        prism._validate_record(
            nn.Linear(self.WIDTH, self.ROWS, bias=False),
            self._record(),
            self._arrays(),
            mx.ones((self.WIDTH,)),
        )

    def test_rejects_kind_mismatch(self):
        with self.assertRaises(ValueError):
            prism._validate_record(
                nn.Linear(self.WIDTH, self.ROWS, bias=False),
                self._record(embedding=True),
                self._arrays(),
                mx.ones((self.WIDTH,)),
            )

    def test_rejects_wrong_packed_shape(self):
        arrays = self._arrays()
        arrays[0] = mx.zeros((self.ROWS, self.WIDTH // 8), dtype=mx.uint32)
        with self.assertRaises(ValueError):
            prism._validate_record(
                nn.Linear(self.WIDTH, self.ROWS, bias=False),
                self._record(),
                arrays,
                mx.ones((self.WIDTH,)),
            )

    def test_rejects_non_unit_sign_vector(self):
        signs = mx.ones((self.WIDTH,)) * 2
        with self.assertRaises(ValueError):
            prism._validate_record(
                nn.Linear(self.WIDTH, self.ROWS, bias=False),
                self._record(),
                self._arrays(),
                signs,
            )

    def test_rejects_missing_sign_vector_when_block_is_set(self):
        with self.assertRaises(ValueError):
            prism._validate_record(
                nn.Linear(self.WIDTH, self.ROWS, bias=False),
                self._record(),
                self._arrays(),
                None,
            )

    def test_rejects_non_finite_scales(self):
        arrays = self._arrays()
        arrays[1] = mx.full((self.ROWS, self.WIDTH // 128), float("nan"), dtype=mx.float16)
        with self.assertRaises(ValueError):
            prism._validate_record(
                nn.Linear(self.WIDTH, self.ROWS, bias=False),
                self._record(),
                arrays,
                mx.ones((self.WIDTH,)),
            )


class SchemaGateTests(unittest.TestCase):
    """Upstream's own runtime rejects everything but schema_version 1, while every
    published Bonsai 2 pack ships 2 — so this gate is a deliberate divergence and is
    worth pinning down."""

    def _pack(self, tmp, **over):
        config = {"model_type": prism.MODEL_TYPE, "schema_version": 2}
        config.update(over)
        (Path(tmp) / "config.json").write_text(json.dumps(config))
        return tmp

    def test_rejects_a_non_prism_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._pack(tmp, model_type="qwen3_5")
            with self.assertRaisesRegex(ValueError, "Not a prism_hadamard_qwen35 pack"):
                prism.load_packed(tmp)

    def test_rejects_an_unknown_schema_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._pack(tmp, schema_version=3)
            with self.assertRaisesRegex(ValueError, "Unsupported packed model schema"):
                prism.load_packed(tmp)

    def test_accepts_both_published_schema_versions(self):
        # Gets past the version gate and fails later, on the absent text_config/weights.
        for version in prism.SUPPORTED_SCHEMA_VERSIONS:
            with tempfile.TemporaryDirectory() as tmp:
                self._pack(tmp, schema_version=version)
                with self.assertRaises(Exception) as caught:
                    prism.load_packed(tmp)
                self.assertNotIn("schema", str(caught.exception).lower())

    def test_is_prism_pack(self):
        self.assertTrue(prism.is_prism_pack({"model_type": prism.MODEL_TYPE}))
        self.assertFalse(prism.is_prism_pack({"model_type": "qwen3_5"}))


class EngineDispatchTests(unittest.TestCase):
    def test_mlx_prism_spec_uses_the_prism_loader(self):
        from mlx_lazyserve import engine

        spec = mock.Mock(engine="mlx_prism", repo="prism-ml/pack")
        with mock.patch.object(engine, "MlxLmModel") as model:
            engine.load_model(spec)
        model.assert_called_once_with("prism-ml/pack", loader=prism.load)

    def test_auto_never_falls_through_to_the_prism_loader(self):
        # "auto" retries with mlx-vlm; routing it to prism instead would turn a
        # legitimately-unloadable repo into a confusing transform error.
        from mlx_lazyserve import engine

        spec = mock.Mock(engine="auto", repo="some/repo")
        with (
            mock.patch.object(engine, "MlxLmModel", side_effect=RuntimeError("nope")),
            mock.patch.object(engine, "MlxVlmModel") as vlm,
        ):
            engine.load_model(spec)
        vlm.assert_called_once_with("some/repo")


if __name__ == "__main__":
    unittest.main()
