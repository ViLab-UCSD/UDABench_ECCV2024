import timm

model_names = [
    "resnet50",
    "deit3_small_patch16_224",
    "convnext_tiny",
    "swinv2_tiny_window16_256",
    "resmlp_24_224"
]

def create_model(model_name, pretrained=True):
    
    model = timm.create_model(model_name, pretrained=pretrained)
    params = sum([p.numel() for p in model.parameters() if p.requires_grad==True])
    params /= 1e6
    # print("Number of parameters in {}: {:.02f}M".format(model_name, params))
    return model, params


def timm_resnet50(pretrained=False, **kwargs):
    model, _ = create_model("resnet50", pretrained=pretrained)
    return model

def timm_deit(pretrained=False, **kwargs):
    model, _ = create_model("deit3_small_patch16_224", pretrained=pretrained)
    return model

def timm_convnext(pretrained=False, **kwargs):
    model, _ = create_model("convnext_tiny", pretrained=pretrained)
    return model

def timm_swin(pretrained=False, **kwargs):
    model, _ = create_model("swinv2_tiny_window16_256", pretrained=pretrained)
    return model

def timm_resmlp(pretrained=False, **kwargs):
    model, _ = create_model("resmlp_24_224", pretrained=pretrained)
    return model