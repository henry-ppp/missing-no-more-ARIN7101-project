"""Training sees only shared incomplete inputs, never evaluation truth."""
from dataclasses import dataclass, asdict
from copy import deepcopy
import hashlib
import os
import random
import time
import numpy as np
import torch
from .dae import DenoisingAutoencoder

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 150
    batch_size: int = 256
    learning_rate: float = 0.001
    training_mask_rate: float = 0.3
    validation_fraction: float = 0.1
    patience: int = 20
    min_delta: float = 0.00001
    hidden_dims: tuple[int, ...] | None = None
    latent_dim: int | None = None
    dropout: float = 0.05
    activation: str = 'relu'
    seed: int = 0
    device: str = 'auto'
    threads: int = 2

    def validate(self):
        if min(self.epochs, self.batch_size, self.patience, self.threads) < 1:
            raise ValueError('epochs, batch size, patience and threads must be positive')
        if not 0 < self.training_mask_rate < 1 or not 0 < self.validation_fraction < 1:
            raise ValueError('Mask and validation fractions must be strictly between 0 and 1')
        if self.learning_rate <= 0 or self.min_delta < 0:
            raise ValueError('Invalid learning rate or minimum improvement')

def split_observed(observed, fraction, seed):
    """Fixed per-column validation cells, absent from all training inputs/targets."""
    rng = np.random.default_rng(seed)
    validation = np.zeros_like(observed, dtype=bool)
    for col in range(observed.shape[1]):
        rows = np.flatnonzero(observed[:, col])
        if len(rows) < 2:
            raise ValueError('Each feature needs at least two observed cells')
        count = min(len(rows) - 1, max(1, int(round(len(rows) * fraction))))
        validation[rng.choice(rows, count, replace=False), col] = True
    return observed & ~validation, validation

def masked_mse(prediction, target, hidden):
    if not hidden.any():
        raise ValueError('No supervised denoising targets in this batch')
    # Index before subtraction: non-target cells never enter the loss expression.
    return (prediction[hidden] - target[hidden]).square().mean()

def corrupt_batch(values, available, rate, generator):
    hidden = (torch.rand(available.shape, generator=generator) < rate) & available.cpu()
    visible = available.cpu() & ~hidden
    inputs = values.clone()
    inputs[~visible] = 0
    return inputs, visible, hidden

def fit(X_missing, config=TrainConfig()):
    config.validate()
    start = time.perf_counter()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if config.device == 'auto' else config.device)
    array = X_missing.to_numpy(dtype=np.float32)
    if np.isinf(array).any() or len(array) < 2 or not X_missing.index.is_unique or not X_missing.columns.is_unique:
        raise ValueError('Expected unique identifiers and numerical observations without infinity')
    observed = np.isfinite(array)
    train_mask, validation_mask = split_observed(observed, config.validation_fraction, config.seed)
    # Validation targets are stored separately and are zeroed out of training values.
    validation_values = torch.from_numpy(np.where(validation_mask, array, 0).astype(np.float32))
    train_values = torch.from_numpy(np.where(train_mask, array, 0).astype(np.float32))
    available = torch.from_numpy(train_mask)
    validation = torch.from_numpy(validation_mask)
    model = DenoisingAutoencoder(array.shape[1], config.hidden_dims, config.latent_dim, config.dropout, config.activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed + 1)
    history, best_state, best_loss, best_epoch, stale = [], None, float('inf'), 0, 0

    def validate():
        model.eval()
        squared_error, count = 0.0, 0
        with torch.no_grad():
            for offset in range(0, len(array), config.batch_size):
                sl = slice(offset, offset + config.batch_size)
                heldout = validation[sl].to(device)
                if not heldout.any():
                    continue
                prediction = model(train_values[sl].to(device), available[sl].to(device))
                target = validation_values[sl].to(device)
                squared_error += (prediction[heldout] - target[heldout]).square().sum().item()
                count += heldout.sum().item()
        return squared_error / count

    initial_validation = validate()
    for epoch in range(1, config.epochs + 1):
        model.train()
        total, count = 0.0, 0
        order = torch.randperm(len(array), generator=generator)
        for offset in range(0, len(array), config.batch_size):
            indices = order[offset:offset + config.batch_size]
            values = train_values[indices]
            inputs, visible, hidden = corrupt_batch(values, available[indices], config.training_mask_rate, generator)
            if not hidden.any():
                continue
            prediction = model(inputs.to(device), visible.to(device))
            loss = masked_mse(prediction, values.to(device), hidden.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            targets = int(hidden.sum())
            total += loss.item() * targets
            count += targets
        if count == 0:
            raise ValueError('No denoising targets drawn in an entire epoch')
        validation_loss = validate()
        if not np.isfinite(validation_loss):
            raise FloatingPointError('Nonfinite validation loss')
        history.append({'epoch': epoch, 'train_masked_mse': total / count, 'validation_masked_mse': validation_loss})
        if validation_loss < best_loss - config.min_delta:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f'epoch={epoch} train={total/count:.6f} validation={validation_loss:.6f}', flush=True)
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    summary = {'config': asdict(config), 'architecture': model.architecture,
        'rows': len(array), 'features': array.shape[1], 'device': str(device),
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
        'epochs_completed': len(history), 'best_epoch': best_epoch, 'best_validation_loss': best_loss,
        'final_validation_loss': history[-1]['validation_masked_mse'],
        'initial_validation_loss': initial_validation, 'runtime_seconds': time.perf_counter() - start,
        'training_available_cells': int(train_mask.sum()), 'validation_cells': int(validation_mask.sum()),
        'validation_mask_sha256': hashlib.sha256(validation_mask.tobytes()).hexdigest(),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'validation_policy': 'Observed-only per-column heldout cells; excluded from all training inputs and targets. Shared scaler unchanged.',
        'objective': 'MSE only on additional masks of non-validation originally observed cells'}
    return model, summary, history, validation_mask
