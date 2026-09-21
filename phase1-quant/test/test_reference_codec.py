#!/usr/bin/env python3
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from reference_codec import (dequantize_adaptive_q2, fwht128, pack2,
                             quantize_adaptive_q2, quantize_symmetric_q4,
                             unpack2)


class ReferenceCodecTest(unittest.TestCase):
    def test_fwht_is_involution(self):
        rng = np.random.default_rng(17)
        value = rng.normal(size=(3, 256)).astype(np.float32)
        restored = fwht128(fwht128(value))
        self.assertLess(np.max(np.abs(restored - value)), 2e-6)

    def test_pack_round_trip(self):
        codes = np.arange(256, dtype=np.uint8).reshape(2, 128) % 4
        np.testing.assert_array_equal(unpack2(pack2(codes)), codes)

    def test_adaptive_never_worse_than_fixed_candidate(self):
        rng = np.random.default_rng(71)
        weight = rng.normal(size=(12, 256)).astype(np.float32)
        weight[:, ::7] *= 4
        importance = np.geomspace(0.05, 20, 256).astype(np.float32)
        fixed = quantize_adaptive_q2(weight, importance, alpha_table=(1 / 3,))
        adaptive = quantize_adaptive_q2(
            weight, importance, alpha_table=(0.2, 0.25, 0.3, 1 / 3, 0.4, 0.5)
        )
        self.assertLessEqual(adaptive.weighted_rel_mse, fixed.weighted_rel_mse + 1e-12)
        reconstructed = dequantize_adaptive_q2(
            adaptive.packed, adaptive.scales, adaptive.alpha_index, adaptive.alpha_table
        )
        self.assertEqual(reconstructed.shape, weight.shape)
        self.assertTrue(np.isfinite(reconstructed).all())
        self.assertGreater(len(np.unique(adaptive.alpha_index)), 1)
        rotated = fwht128(weight)
        h = np.broadcast_to(
            importance.reshape(1, 2, 128), (weight.shape[0], 2, 128)
        ).reshape(weight.shape)
        stored_error = np.sum(h * np.square(rotated - reconstructed), dtype=np.float64)
        self.assertAlmostEqual(adaptive.weighted_error, stored_error, places=4)

    def test_q4_improves_weighted_error(self):
        rng = np.random.default_rng(711)
        weight = rng.standard_t(df=3, size=(16, 128)).astype(np.float32)
        importance = rng.lognormal(size=128).astype(np.float32)
        q2 = quantize_adaptive_q2(weight, importance)
        _, _, q4_rel = quantize_symmetric_q4(fwht128(weight), importance)
        self.assertLess(q4_rel, q2.weighted_rel_mse)


if __name__ == "__main__":
    unittest.main()
