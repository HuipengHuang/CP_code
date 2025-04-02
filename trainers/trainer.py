import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import models
from predictors import predictor
from loss.utils import get_loss_function
from .cadapter import CAdapter, Adapter
from .early_stoping import EarlyStopping

class Trainer:
    """
    Trainer class that implement all the functions regarding training.
    All the arguments are passed through args."""
    def __init__(self, args, num_classes):
        self.device = torch.device(f"cuda:{args.gpu}")
        self.net = models.utils.build_model(args, num_classes=num_classes)
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
            self.predictor = predictor.Predictor(args, self.net, num_classes=num_classes,
                                                 adapter=self.adapter,
                                                 final_activation_function=final_activation_function)
        elif args.adapter == "True":
            input_feature = models.utils.get_model_output_dim(args, self.net)
            self.adapter = Adapter(input_feature, num_classes, self.device)
            self.set_train_mode((args.train_net == "True"), (args.train_adapter == "True"))
            self.predictor = predictor.Predictor(args, self.net, num_classes=num_classes,
                                                 adapter=self.adapter,
                                                 final_activation_function=final_activation_function)
        else:
            self.adapter = None
            self.predictor = predictor.Predictor(args, self.net, num_classes,
                                                 final_activation_function=final_activation_function)
        self.predictor.set_mode("train")
        self.num_classes = num_classes
        self.loss_function = get_loss_function(args, self.predictor)

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

    def train(self, data_loader, epochs):
        self.net.train()
        for epoch in range(epochs):
            for data, target in tqdm(data_loader, desc=f"Epoch: {epoch+1} / {epochs}"):
                self.train_batch(data, target)

            if self.scheduler:
                self.scheduler.step()


    def set_train_mode(self, train_net, train_adapter):
        assert self.adapter is not None, print("The trainer does not have an adapter.")
        if not train_adapter:
            for param in self.adapter.adapter_net.parameters():
                param.requires_grad = train_adapter
        if not train_net:
            for param in self.net.parameters():
                param.requires_grad = train_net
