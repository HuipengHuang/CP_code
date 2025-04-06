import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import models
from predictors.utils import get_predictor
from loss.utils import get_loss_function
from .cadapter import CAdapter, Adapter
from .early_stopping import EarlyStopping

class Trainer:
    """
    Trainer class that implement all the functions regarding training.
    All the arguments are passed through args."""
    def __init__(self, args, num_classes):
        self.device = torch.device(f"cuda:{args.gpu}")
        self.net = models.get_model.build_model(args, num_classes=num_classes)
        self.batch_size = args.batch_size
        if args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                             weight_decay=args.weight_decay, nesterov=(args.nesterov == "True"))
        if args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)

        if args.learning_rate_scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=200)
        else:
            self.scheduler = None

        final_activation_function = args.final_activation_function

        if args.cadapter == "True":
            self.adapter = CAdapter(num_classes, num_classes, self.device)
            self.set_train_mode((args.train_net == "True"), (args.train_adapter == "True"))
            self.predictor = get_predictor(args, self.net, num_classes=num_classes,
                                                 adapter=self.adapter,
                                                 final_activation_function=final_activation_function)

        elif args.adapter == "True":
            input_feature = models.get_model.get_model_output_dim(args, self.net)
            self.adapter = Adapter(input_feature, num_classes, self.device)
            self.set_train_mode((args.train_net == "True"), (args.train_adapter == "True"))
            self.predictor = get_predictor(args, self.net, num_classes=num_classes,
                                                 adapter=self.adapter,
                                                 final_activation_function=final_activation_function)
        else:
            self.adapter = None
            self.predictor = get_predictor(args, self.net, num_classes,
                                                 final_activation_function=final_activation_function)
        self.predictor.set_mode("train")
        self.num_classes = num_classes
        self.loss_function = get_loss_function(args, self.predictor)
        if args.patience:
            self.early_stopping = EarlyStopping(patience=args.patience)
        else:
            self.early_stopping = None

    def train_batch(self, data, target):
        data = data.to(self.device)
        target = target.to(self.device)

        logits = self.net(data)
        if self.adapter:
            logits = self.adapter(logits)

        loss = self.loss_function(logits, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self, train_loader, epochs, val_loader=None):
        self.net.train()
        if val_loader is None or self.early_stopping is None:
            for epoch in range(epochs):
                for data, target in tqdm(train_loader, desc=f"Epoch: {epoch + 1} / {epochs}"):
                    self.train_batch(data, target)

                if self.scheduler:
                    self.scheduler.step()
        else:
            for epoch in range(epochs):
                for data, target in tqdm(train_loader, desc=f"Epoch: {epoch + 1} / {epochs}"):
                    self.train_batch(data, target)

                val_loss = self.compute_validation_loss(val_loader)
                stop = self.early_stopping(val_loss, epoch)
                if stop:
                    break
                if self.scheduler:
                    self.scheduler.step()

    def compute_validation_loss(self, val_loader):
        loss = 0
        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)
            logits = self.net(data)
            loss += self.loss_function(logits, target).item()
        return loss / len(val_loader)

    def set_train_mode(self, train_net, train_adapter):
        assert self.adapter is not None, print("The trainer does not have an adapter.")
        if not train_adapter:
            for param in self.adapter.adapter_net.parameters():
                param.requires_grad = train_adapter
        if not train_net:
            for param in self.net.parameters():
                param.requires_grad = train_net
