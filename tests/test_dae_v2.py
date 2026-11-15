import unittest
import numpy as np
import pandas as pd
import torch
from models.deep_learning_v2.dae import DenoisingAutoencoder
from models.deep_learning_v2.train import TrainConfig, corrupt_batch, fit, masked_mse, split_observed
from models.deep_learning_v2.predict import predict


class DAETests(unittest.TestCase):
    def test_shapes(self):
        for features in (8, 11, 14, 81):
            model = DenoisingAutoencoder(features)
            values = torch.zeros(3, features)
            self.assertEqual(model(values, values.bool()).shape, values.shape)

    def test_exclusive_mask_and_gradient(self):
        observed = np.ones((12, 8), dtype=bool)
        observed[0, 0] = False
        training, validation = split_observed(observed, 0.2, 0)
        np.testing.assert_array_equal(training | validation, observed)
        self.assertFalse((training & validation).any())
        available = torch.from_numpy(training)
        targets = torch.ones(12, 8)
        inputs, visible, hidden = corrupt_batch(targets, available, 0.5, torch.Generator().manual_seed(4))
        self.assertTrue(hidden.any())
        self.assertFalse((hidden & ~available).any())
        self.assertTrue(torch.all(inputs[~visible] == 0))
        predictions = torch.zeros_like(targets, requires_grad=True)
        targets[~hidden] = float('nan')
        loss = masked_mse(predictions, targets, hidden)
        self.assertEqual(loss.item(), 1)
        loss.backward()
        self.assertTrue(torch.all(predictions.grad[~hidden] == 0))
        self.assertTrue(torch.all(predictions.grad[hidden] != 0))

    def test_fit_updates_and_preserves_input(self):
        frame = pd.DataFrame(np.random.default_rng(2).normal(size=(24, 8)))
        frame.iloc[::3, 1] = np.nan
        config = TrainConfig(epochs=2, batch_size=8, device='cpu')
        torch.manual_seed(config.seed)
        initial = DenoisingAutoencoder(8)
        model, summary, history, validation = fit(frame, config)
        self.assertEqual(len(history), 2)
        self.assertFalse((validation & frame.isna().to_numpy()).any())
        self.assertTrue(any(not torch.equal(before, after) for before, after in zip(initial.parameters(), model.parameters())))
        result = predict(model, frame)
        pd.testing.assert_index_equal(result.index, frame.index)
        pd.testing.assert_index_equal(result.columns, frame.columns)
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        observed = frame.notna().to_numpy()
        np.testing.assert_array_equal(result.to_numpy()[observed], frame.to_numpy()[observed])
