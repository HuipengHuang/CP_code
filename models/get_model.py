import torch
import torch.nn as nn
import torchvision.models as models
from . import attention_base_model, transmil, rrtmil, dsmil


def build_mil_model(args, num_classes):
    model_type = args.model
    device = torch.device(f"cuda:{args.gpu}")
    net = None

    if model_type == "attention":
        net = attention_base_model.AttentionModel(input_dim=args.input_dimension)

    elif model_type == "gradattention":
        net = attention_base_model.GatedAttentionModel(input_dim=args.input_dimension)

    elif model_type == "transmil":
        if args.final_activation_function != "sigmoid":
            print(f"Attention. Activation function you use for the binary classification task is {args.final_activation_function} but not sigmoid.")
        assert num_classes is not None, print("num_classes is none")
        net = transmil.TransMIL(device, input_dim=args.input_dimension, n_classes=num_classes)

    elif model_type == "rrtmil":
        net = rrtmil.RRT(input_dim=args.input_dimension)

    elif model_type == "dsmil":
        i_classifier = dsmil.FCLayer(in_size=args.input_dimension, out_size=num_classes).to(device)
        b_classifier = dsmil.BClassifier(input_size=args.input_dimension, mDim=args.input_dimension, output_class=num_classes,
                                       passing_v=False).to(device)
        net = dsmil.MILNet(i_classifier, b_classifier).to(device)

    elif net is None:
        raise ValueError(f"Unsupported model type: {model_type}")
    return net

def build_model(args, num_classes=None):
    model_type = args.model
    device = torch.device(f"cuda:{args.gpu}")
    pretrained = (args.pretrained == "True")
    net = None

    if args.multi_instance_learning:
        net = build_mil_model(args, num_classes)
        return net.to(device)

    if model_type == "resnet34":
        net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnet101":
        net = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "densenet121":
        net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "densenet161":
        net = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnext50":
        net = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnext101":
        net = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1 if pretrained else None)

    if hasattr(net, "fc") and num_classes:
        #  ResNet and ResNeXt
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    elif hasattr(net, "classifier") and num_classes:
        #  DenseNet
        net.classifier = torch.nn.Linear(net.classifier.in_features, num_classes)


    if net is None:
        raise ValueError(f"Unsupported model type: {model_type}")

    return net.to(device)

def get_model_output_dim(args, net):
    if args.model == "attention" or args.model == "gradattention":
        output_feature = 2
    elif hasattr(net, "fc"):
        #  ResNet and ResNeXt
        output_feature = net.fc.out_features
    elif hasattr(net, "classifier"):
        #  DenseNet
        output_feature = net.classifier.out_features
    return output_feature

def load_model(args, num_classes, device, path):
    model_type = args.model
    pretrained = (args.pretrained == "True")
    net = None
    if model_type == "resnet34":
        net = models.resnet34()
    elif model_type == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnet101":
        net = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "densenet121":
        net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "densenet161":
        net = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnext50":
        net = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnext101":
        net = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1 if pretrained else None)

    if hasattr(net, "fc"):
        #  ResNet and ResNeXt
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    elif hasattr(net, "classifier"):
        #  DenseNet
        net.classifier = torch.nn.Linear(net.classifier.in_features, num_classes)

    if model_type == "attention":
        net = attention_base_model.AttentionModel()
    elif model_type == "gradattention":
        net = attention_base_model.GatedAttentionModel()
    elif net is None:
        raise ValueError(f"Unsupported model type: {model_type}")
    net = net.load_state_dict(torch.load(path))
    return net.to(device)



