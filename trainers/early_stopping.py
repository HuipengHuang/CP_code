import torch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, stop_epoch=50):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = torch.inf

    def __call__(self, val_loss, epoch):
        if val_loss < self.best_loss:
            self.counter += 1
            self.best_loss = val_loss
            if self.counter >= self.patience and epoch >= self.stop_epoch:
                return True
        return False