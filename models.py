import timm


def load_model(model_name: str, pretrained: bool = True, num_classes: int = 1000):
    model_name = model_name.lower()

    model = None
    
    if model_name == 'resnet18':
        model = timm.create_model(
            'resnet18',
            pretrained=pretrained,
            num_classes=num_classes
        )
    elif model_name == 'resnet50':
        model = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            num_classes=num_classes
        )
    elif model_name == 'resnest14d':
        model = timm.create_model(
            'resnest14d',
            pretrained=pretrained,
            num_classes=num_classes
        )
    elif model_name == 'resnest26d':
        model = timm.create_model(
            'resnest26d',
            pretrained=pretrained,
            num_classes=num_classes
        )
    elif model_name == 'mlp':
        model = timm.create_model(
            'mixer_b16_224',
            pretrained=pretrained,
            num_classes=num_classes
        )
    
    return model
