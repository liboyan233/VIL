# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

from typing import Optional
import IPython
import copy
from warnings import warn
e = IPython.embed
vit_flag = False  # TODO, implement it wisely

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        self.swin = False
        self.vit = False
        print("Backbone:", backbone.__class__.__name__)
        if backbone.__class__.__name__ == 'SwinTransformer':
            return_layers = {"features": "0"}
            self.swin = True
        elif backbone.__class__.__name__ == 'VisionTransformer' or backbone.__class__.__name__ == 'DinoVisionTransformer':
            print("this is a ViT backbone")
            # backbone.heads = torch.nn.Identity()  # remove classification head
            return_layers = None
            self.vit = True
            global vit_flag
            vit_flag = True

        elif return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        if return_layers is not None:
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        else:
            self.body = backbone
        self.num_channels = num_channels
        if self.swin:
            self.ada_feature = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=1) # TODO do we want this?
        elif self.vit:
            self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))  # for ViT
            self.ada_feature = nn.Sequential(
                nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=1),  # Downsamples and projects
                nn.ReLU()
            )

    def forward(self, tensor):
        try:
            features = self.body.forward_features(tensor)  # for ViT, use forward_features
            patch_tokens = features["x_norm_patchtokens"]  # [1, 256, 384]
            features = patch_tokens.permute(0, 2, 1).reshape(-1, 384, 16, 16)
            xs = self.ada_feature(features)  # [N, C, H, W]
            xs = self.avg_pool(xs)  # [N, C, 7, 7] for ViT
            warn(' using vit branch ')

            return xs  # [N, C, H, W]
        except:
            xs = self.body(tensor)
            # print("swin:", self.swin, " vit:", self.vit)
            # print("vit output shape:", xs.shape)
            # xs = self.ada_feature(xs.permute(0, 3, 1, 2))  # N,C,_,_
            if self.swin:
                for name, x in xs.items():
                    xs[name] = self.ada_feature(x.permute(0, 3, 1, 2))
            # elif self.vit:
           
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out

# class SwinWrapper(nn.Module):
#     def __init__(self, backbone: nn.Module, input_channels, output_channels: int):
#         super().__init__()
#         self.body = backbone
#         self.num_channels = output_channels
#         if input_channels != output_channels:
#             self.ada_feature = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1)
#         else:
#             self.ada_feature = nn.Identity()

#     def forward(self, tensor):
#         xs = self.body(tensor)
#         xs = self.ada_feature(xs)
#         return xs

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: Optional[bool] = False,
                 swin_local_ckpt: Optional[str] = None,
                 freeze_backbone: Optional[bool] = False):

        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        if name == 'swin_tiny':
            num_channels = 784
            # print("=== Swin Transformer Tiny Backbone ===")
        elif name == 'vit':
            num_channels = 384
            # print("=== ViT Backbone ===")

        if name == 'swin_tiny':
            warn("=== Using Swin Transformer Tiny Backbone ===")
            if swin_local_ckpt == "default":
                warn("=== Loading default checkpoint ===")
                backbone = getattr(torchvision.models, 'swin_t')(weights='DEFAULT')
                # backbone = SwinWrapper(backbone, num_channels, 512)
            elif swin_local_ckpt is not None and swin_local_ckpt != "None":
                warn(f"=== Loading checkpoint from {swin_local_ckpt} ===")
                backbone = getattr(torchvision.models, 'swin_t')(weights=None)
                miss_key, unexpected_key = load_swint(swin_local_ckpt, backbone)
                warn(f"=== Checkpoint loaded with {len(miss_key)} missing keys and {len(unexpected_key)} unexpected keys. ===")
                # backbone = SwinWrapper(backbone, num_channels, 512)
            else:
                warn("=== Traing from scratch! ===")
                backbone = getattr(torchvision.models, 'swin_t')(weights=None)
                # backbone = SwinWrapper(backbone, num_channels, 512)
            if freeze_backbone:
                for _, parameter in backbone.named_parameters():
                    parameter.requires_grad_(False)
                warn("=== Freezing backbone parameters ===")
            else:
                warn("=== Updating backbone parameters ===")
        elif 'vit' in name:
            warn("=== Using ViT Backbone ===")
            backbone = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vits14')
            # backbone = getattr(torchvision.models, name)(weights='DEFAULT')
            if freeze_backbone:
                for _, parameter in backbone.named_parameters():
                    parameter.requires_grad_(False)
                warn("=== Freezing backbone parameters ===")
            else:
                warn("=== Updating backbone parameters ===")
        else:
            warn(f"=== Using {name} Backbone ===")
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
            if freeze_backbone:
                for _, parameter in backbone.named_parameters():
                    parameter.requires_grad_(False)
                warn("=== Freezing backbone parameters ===")
        

        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        # print(f"Using backbone {self[0].body.__class__.__name__} with position encoding {self[1].__class__.__name__}")
        if vit_flag:
            tensor_list = F.interpolate(
                tensor_list,
                size=(224, 224),
                mode='bilinear',  # or 'bicubic', 'nearest'
                align_corners=False  # Set to True if using 'bilinear'/'bicubic' for legacy behavior
            )
            warn("=== interpolating input for ViT backbone ===")
        xs = self[0](tensor_list)  # take the first element for nn.Sequential, i.e. backbone
        # if isinstance(xs, (list, tuple)):
        #     print("Backbone output is a list or tuple, converting to dict")
        # elif isinstance(xs, torch.Tensor):
        #     print("shape of xs:", xs.shape)
        out: List[NestedTensor] = []
        pos = []
        try:  # 
            for name, x in xs.items():
                out.append(x)  # x shape: [N, C, H, W]
                print('backbone output shape:', x.shape, 'with backbone', self[0].body.__class__.__name__)
                # position encoding
                pos.append(self[1](x).to(x.dtype))
        except:
            out.append(xs)  # should also be in shape [N, C, H, W]
            print('backbone output shape1:', xs.shape)

            pos.append(self[1](xs).to(xs.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, args.swin_local_ckpt, args.freeze_backbone)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def load_swint(ckpt_path, backbone):

    key_mapping = {
        "encoder.patch_embed.proj": "features.0.0",
        "encoder.patch_embed.norm": "features.0.2",
        "encoder.layers.0.blocks.0": "features.1.0",
        "encoder.layers.0.blocks.1": "features.1.1",
        "encoder.layers.0.downsample": "features.2",
        "encoder.layers.1.blocks.0": "features.3.0",
        "encoder.layers.1.blocks.1": "features.3.1",
        "encoder.layers.1.downsample": "features.4",
        "encoder.layers.2.blocks.0": "features.5.0",
        "encoder.layers.2.blocks.1": "features.5.1",
        "encoder.layers.2.blocks.2": "features.5.2",
        "encoder.layers.2.blocks.3": "features.5.3",
        "encoder.layers.2.blocks.4": "features.5.4",
        "encoder.layers.2.blocks.5": "features.5.5",
        "encoder.layers.2.downsample": "features.6",
        "encoder.layers.3.blocks.0": "features.7.0",
        "encoder.layers.3.blocks.1": "features.7.1",
        "encoder.norm": "norm"
    }

    specified = {
        "encoder.layers.0.blocks.0.mlp.fc1.weight": "features.1.0.mlp.0.weight",
        "encoder.layers.0.blocks.0.mlp.fc1.bias": "features.1.0.mlp.0.bias",
        "encoder.layers.0.blocks.0.mlp.fc2.weight": "features.1.0.mlp.3.weight",
        "encoder.layers.0.blocks.0.mlp.fc2.bias": "features.1.0.mlp.3.bias",
        "encoder.layers.0.blocks.1.mlp.fc1.weight": "features.1.1.mlp.0.weight",
        "encoder.layers.0.blocks.1.mlp.fc1.bias": "features.1.1.mlp.0.bias",
        "encoder.layers.0.blocks.1.mlp.fc2.weight": "features.1.1.mlp.3.weight",
        "encoder.layers.0.blocks.1.mlp.fc2.bias": "features.1.1.mlp.3.bias",
        'encoder.layers.1.blocks.0.mlp.fc1.weight': "features.3.0.mlp.0.weight", 
        'encoder.layers.1.blocks.0.mlp.fc1.bias': "features.3.0.mlp.0.bias", 
        'encoder.layers.1.blocks.0.mlp.fc2.weight': "features.3.0.mlp.3.weight", 
        'encoder.layers.1.blocks.0.mlp.fc2.bias': "features.3.0.mlp.3.bias",
        'encoder.layers.1.blocks.1.mlp.fc1.weight': "features.3.1.mlp.0.weight", 
        'encoder.layers.1.blocks.1.mlp.fc1.bias': "features.3.1.mlp.0.bias", 
        'encoder.layers.1.blocks.1.mlp.fc2.weight': "features.3.1.mlp.3.weight", 
        'encoder.layers.1.blocks.1.mlp.fc2.bias': "features.3.1.mlp.3.bias",
        'encoder.layers.2.blocks.0.mlp.fc1.weight': "features.5.0.mlp.0.weight", 
        'encoder.layers.2.blocks.0.mlp.fc1.bias': "features.5.0.mlp.0.bias", 
        'encoder.layers.2.blocks.0.mlp.fc2.weight': "features.5.0.mlp.3.weight", 
        'encoder.layers.2.blocks.0.mlp.fc2.bias': "features.5.0.mlp.3.bias", 
        'encoder.layers.2.blocks.1.mlp.fc1.weight': "features.5.1.mlp.0.weight", 
        'encoder.layers.2.blocks.1.mlp.fc1.bias': "features.5.1.mlp.0.bias", 
        'encoder.layers.2.blocks.1.mlp.fc2.weight': "features.5.1.mlp.3.weight", 
        'encoder.layers.2.blocks.1.mlp.fc2.bias': "features.5.1.mlp.3.bias",
        'encoder.layers.2.blocks.2.mlp.fc1.weight': "features.5.2.mlp.0.weight", 
        'encoder.layers.2.blocks.2.mlp.fc1.bias': "features.5.2.mlp.0.bias", 
        'encoder.layers.2.blocks.2.mlp.fc2.weight': "features.5.2.mlp.3.weight", 
        'encoder.layers.2.blocks.2.mlp.fc2.bias': "features.5.2.mlp.3.bias",

        'encoder.layers.2.blocks.3.mlp.fc1.weight': "features.5.3.mlp.0.weight",
        'encoder.layers.2.blocks.3.mlp.fc1.bias': "features.5.3.mlp.0.bias",
        'encoder.layers.2.blocks.3.mlp.fc2.weight': "features.5.3.mlp.3.weight",
        'encoder.layers.2.blocks.3.mlp.fc2.bias': "features.5.3.mlp.3.bias",

        'encoder.layers.2.blocks.4.mlp.fc1.weight': "features.5.4.mlp.0.weight",
        'encoder.layers.2.blocks.4.mlp.fc1.bias': "features.5.4.mlp.0.bias",
        'encoder.layers.2.blocks.4.mlp.fc2.weight': "features.5.4.mlp.3.weight",
        'encoder.layers.2.blocks.4.mlp.fc2.bias': "features.5.4.mlp.3.bias",

        'encoder.layers.2.blocks.5.mlp.fc1.weight': "features.5.5.mlp.0.weight",
        'encoder.layers.2.blocks.5.mlp.fc1.bias': "features.5.5.mlp.0.bias",
        'encoder.layers.2.blocks.5.mlp.fc2.weight': "features.5.5.mlp.3.weight",
        'encoder.layers.2.blocks.5.mlp.fc2.bias': "features.5.5.mlp.3.bias",

        'encoder.layers.3.blocks.0.mlp.fc1.weight': "features.7.0.mlp.0.weight",
        'encoder.layers.3.blocks.0.mlp.fc1.bias': "features.7.0.mlp.0.bias",
        'encoder.layers.3.blocks.0.mlp.fc2.weight': "features.7.0.mlp.3.weight",
        'encoder.layers.3.blocks.0.mlp.fc2.bias': "features.7.0.mlp.3.bias",

        'encoder.layers.3.blocks.1.mlp.fc1.weight': "features.7.1.mlp.0.weight",
        'encoder.layers.3.blocks.1.mlp.fc1.bias': "features.7.1.mlp.0.bias",
        'encoder.layers.3.blocks.1.mlp.fc2.weight': "features.7.1.mlp.3.weight",
        'encoder.layers.3.blocks.1.mlp.fc2.bias': "features.7.1.mlp.3.bias",
        
    }

    state_dict = torch.load(ckpt_path)
    param_dict = state_dict['model']

    # May need some adjustions on key names
    copy_ckpt = copy.deepcopy(param_dict)
    for k, _ in param_dict.items():
        if not k.startswith("encoder."):
            copy_ckpt.pop(k) 
        elif k.startswith('predictor'):
            copy_ckpt.pop(k)
        elif 'attn_mask' in k:
            copy_ckpt.pop(k)
        elif 'queue' in k:
            copy_ckpt.pop(k)
        else:
            continue
        # TODO add key mappings 
    # print('existed keys:', state_dict['model'].keys())
    map_keys = list(key_mapping.keys()) 
    ada_ckpt = OrderedDict()
    for k, v in copy_ckpt.items():
        if 'relative_position_index' in k:
            v = v.reshape(-1)  
        if k in specified:
            ada_ckpt[specified[k]] = v
            # print(f"Mapping {k} to {specified[k]}")
        else:
            for map_key in map_keys:
                if map_key in k:
                    new_key = k.replace(map_key, key_mapping[map_key])
                    ada_ckpt[new_key] = v
                    # print(f"Mapping {k} to {new_key}")
                    break
    
    miss_key, unexpected_key = backbone.load_state_dict(ada_ckpt, strict=False)
    for name, module in backbone.named_modules(): # TODO whether to freeze norm layers
        if isinstance(module, torch.nn.LayerNorm):
            for param in module.parameters():
                param.requires_grad = False
    # print('miss_key:', miss_key)
    # print('unexpected_key:', unexpected_key)
    # not for classification . instead adapt the feature dim
    return miss_key, unexpected_key