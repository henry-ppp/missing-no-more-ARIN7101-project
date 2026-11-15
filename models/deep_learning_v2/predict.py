"""Inference uses original incomplete input; observed cells are preserved exactly."""
import numpy as np
import torch
from .dae import DenoisingAutoencoder

def predict(model, X_missing, batch_size=1024, device='cpu'):
    if list(X_missing.columns) != getattr(model, 'feature_names', list(X_missing.columns)):
        raise ValueError('Prediction feature order differs from training')
    if X_missing.shape[1] != model.architecture['n_features'] or batch_size < 1:
        raise ValueError('Invalid prediction shape or batch size')
    array = X_missing.to_numpy(dtype=np.float32)
    if np.isinf(array).any():
        raise ValueError('Infinite input values')
    observed = np.isfinite(array)
    values = np.where(observed, array, 0).astype(np.float32)
    model = model.to(device).eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(array), batch_size):
            sl = slice(start, start + batch_size)
            outputs.append(model(torch.from_numpy(values[sl]).to(device),
                                 torch.from_numpy(observed[sl]).to(device)).cpu().numpy())
    predictions = np.concatenate(outputs)
    if not np.isfinite(predictions[~observed]).all():
        raise FloatingPointError('Nonfinite missing-cell predictions')
    result = X_missing.copy()
    result[:] = np.where(observed, X_missing.to_numpy(), predictions)
    return result

def save_checkpoint(path, model, columns, summary):
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({'state_dict': state, 'architecture': model.architecture,
                'columns': list(columns), 'summary': summary}, path)

def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = DenoisingAutoencoder(**checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'])
    model.feature_names = checkpoint['columns']
    return model.eval(), checkpoint['summary']
