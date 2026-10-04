import unittest

import torch

from mssa_transformer.transformer.transformer_imputation_mssa_mlp_re_bias import (
    SpectralAwareTransformerImputerMSSAMLP,
)


class AblationVariantTests(unittest.TestCase):
    def test_without_transformer_keeps_mssa_modal_fusion_and_uses_mlp_head(self):
        model = SpectralAwareTransformerImputerMSSAMLP(
            input_dim=5,
            mssa_feature_dim=15,
            n_modes=3,
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=32,
            mlp_hidden_dims=[16],
            use_modal_fusion=True,
            use_spectral_bias=True,
            use_transformer_encoder=False,
        )
        source = torch.randn(2, 12, 5)
        source_mask = torch.ones_like(source)
        mssa = torch.randn(2, 12, 5, 3)

        output = model(source, source_mask, mssa)

        self.assertEqual(tuple(output.shape), (2, 1, 5))
        self.assertTrue(model.use_mssa)
        self.assertTrue(model.use_modal_fusion)
        self.assertFalse(model.use_transformer_encoder)
        self.assertFalse(model.use_spectral_bias)
        self.assertTrue(hasattr(model, "reconstruction_head"))
        self.assertFalse(hasattr(model, "encoder_layers"))

    def test_without_mssa_retains_transformer_encoder(self):
        model = SpectralAwareTransformerImputerMSSAMLP(
            input_dim=5,
            mssa_feature_dim=None,
            n_modes=None,
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=32,
            use_spectral_bias=True,
            use_transformer_encoder=True,
        )
        source = torch.randn(2, 12, 5)
        output = model(source, torch.ones_like(source), None)

        self.assertEqual(tuple(output.shape), (2, 1, 5))
        self.assertFalse(model.use_mssa)
        self.assertTrue(model.use_transformer_encoder)
        self.assertTrue(model.use_spectral_bias)


if __name__ == "__main__":
    unittest.main()
