import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import AUROC
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
        self.args = args

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

    def train_loop(self, train_loader, epoch):
            self.net.train()
            for data, target in tqdm(train_loader, desc=f"Epoch: {epoch + 1} / {self.args.epochs}"):
                self.train_batch(data, target)

            if self.scheduler:
                self.scheduler.step()

    def val_loop(self, val_loader):
        self.net.eval()
        loss = 0.
        bag_logit, bag_labels = [], []

        with torch.no_grad():
            for i, data, target in enumerate(val_loader):
                bag_labels.append(target.item())
                data = data.to(self.device)
                label = target.to(self.device)

                test_logits = self.net(data)

                test_loss = self.loss_function(test_logits, label)
                loss += test_loss.item()
                bag_logit.append(torch.softmax(test_logits, dim=-1)[:, 1].cpu().squeeze().numpy())


            accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_logit,)
            loss = loss / len(val_loader.datasets)
            print(f"accuracy:{accuracy}, auc:{auc_value}, precision:{precision}, recall:{recall}, fscore:{fscore}, loss:{loss}")
            return accuracy, auc_value, precision, recall, fscore, loss

    def train(self, train_loader, epochs, val_loader=None):
        self.net.train()
        if val_loader is None or self.early_stopping is None:
            for epoch in range(epochs):
                self.train_loop(train_loader, epoch)
        else:
            for epoch in range(epochs):
                self.train_loop(train_loader, epoch)
                accuracy, auc_value, precision, recall, fscore, loss = self.val_loop(val_loader)
                stop = self.early_stopping(loss, epoch)
                if stop:
                    break



    def set_train_mode(self, train_net, train_adapter):
        assert self.adapter is not None, print("The trainer does not have an adapter.")
        if not train_adapter:
            for param in self.adapter.adapter_net.parameters():
                param.requires_grad = train_adapter
        if not train_net:
            for param in self.net.parameters():
                param.requires_grad = train_net




"""    def val_loop(self, val_loader):
        self.net.eval()
        loss = 0
        accuracy = 0
        with torch.no_grad():
            label = torch.tensor([], device=self.device)
            positive_class_prob = torch.tensor([], device=self.device)
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                logits = self.net(data)

                val_loss = self.loss_function(logits, target)
                loss += val_loss.item()

                prob = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(prob, dim=-1)
                accuracy += (prediction == target).sum().item()

                positive_class_prob = torch.cat((positive_class_prob, prob[:, 1]), dim=0)
                label = torch.cat((label, target), dim=0)
            loss = loss / len(val_loader.dataset)
            accuracy = accuracy / len(val_loader.dataset)
            auroc = AUROC("binary")
            auc = auroc(positive_class_prob, label)
            print(f"Accuracy:{accuracy}, AUC: {auc} loss: {loss}")
            return loss"""