import numpy as np
import torch

class EarlyStopping:
    """Early stops training if validation loss doesn't improve after a given patience, GPU-based."""
    def __init__(self, device, patience=30, delta=0.0, verbose=False):
        self.patience = patience
        self.delta = torch.tensor(delta, dtype=torch.float32, device=device)  # GPU tensor
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = torch.tensor(float('inf'), dtype=torch.float32, device=device)
        self.best_state_dict = None
        self.device = device

    def __call__(self, val_loss, model):
        # Ensure val_loss is a tensor on GPU
        if not isinstance(val_loss, torch.Tensor):
            val_loss = torch.tensor(val_loss, dtype=torch.float32, device=self.device)
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.update_best_state(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.update_best_state(val_loss, model)
            self.counter = 0

    def update_best_state(self, val_loss, model):
        """Updates the best model state in memory on GPU."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min.item():.6f} → {val_loss.item():.6f}). Updating best model state...')
        # Keep state dict on GPU with a deep copy
        self.best_state_dict = {k: v.clone().to(self.device) for k, v in model.state_dict().items()}
        self.val_loss_min = val_loss.clone().to(self.device)