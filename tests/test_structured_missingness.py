import unittest

import numpy as np
import pandas as pd

from mssa_transformer.structured_missingness import (
    DATASET_SPECS,
    apply_structured_missing,
    generate_block_rows,
)


class StructuredMissingnessTests(unittest.TestCase):
    def test_block_rows_have_exact_requested_count(self):
        mask, blocks = generate_block_rows(101, missing_rate=0.4, n_blocks=4, seed=7)
        self.assertEqual(mask.sum(), 40)
        self.assertEqual(len(blocks), 4)
        for start, end in blocks:
            self.assertTrue(mask[start:end].all())
        for (_, left_end), (right_start, _) in zip(blocks, blocks[1:]):
            self.assertGreaterEqual(right_start - left_end, 1)

    def test_all_group_columns_share_mask_and_protected_columns_do_not(self):
        spec = DATASET_SPECS["swat"]
        columns = [spec.index_column, spec.label_column]
        for pattern in spec.patterns.values():
            columns.extend(pattern.columns)
        columns = list(dict.fromkeys(columns))
        frame = pd.DataFrame(
            np.arange(100 * len(columns), dtype=float).reshape(100, len(columns)),
            columns=columns,
        )

        masked, mask, metadata = apply_structured_missing(
            frame, "swat", "sensor_actuator", missing_rate=0.4, n_blocks=4, seed=2
        )
        selected = list(spec.patterns["sensor_actuator"].columns)
        reference = mask[selected[0]].to_numpy()
        for column in selected[1:]:
            np.testing.assert_array_equal(reference, mask[column].to_numpy())
        self.assertEqual(reference.sum(), 40)
        self.assertFalse(mask[spec.index_column].any())
        self.assertFalse(mask[spec.label_column].any())
        self.assertFalse(masked.loc[:, selected].notna().to_numpy()[reference].any())
        self.assertAlmostEqual(metadata["actual_group_missing_rate"], 0.4)

    def test_seed_is_reproducible(self):
        first, first_blocks = generate_block_rows(250, seed=123)
        second, second_blocks = generate_block_rows(250, seed=123)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_blocks, second_blocks)


if __name__ == "__main__":
    unittest.main()
