import torch
import torchvision.models as tvm


def set_requires_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad = requires_grad


def create_model(model_name: str, num_classes: int, pretrained: bool = True, feature_extract: bool = False):
    model_name = model_name.lower()
    if model_name == "resnet18":
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif model_name == "resnet34":
        model = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif model_name == "vgg16":
        model = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        classifier = list(model.classifier)
        classifier[-1] = torch.nn.Linear(in_features, num_classes)
        model.classifier = torch.nn.Sequential(*classifier)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    if feature_extract:
        set_requires_grad(model, False)
        # 只训练最后分类层
        for p in model.parameters():
            p.requires_grad = False
        if model_name.startswith("resnet"):
            for p in model.fc.parameters():
                p.requires_grad = True
        else:
            for p in model.classifier.parameters():
                p.requires_grad = True

    return model