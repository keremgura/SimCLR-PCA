import torch.nn as nn
import torchvision.models as models
from transformers import ViTConfig, ViTModel
import torch.nn.functional as F
import torch
import timm
from torch.nn import SyncBatchNorm
from torch.nn import BatchNorm1d

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim, dropout=0.0):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=False),
            "resnet50": models.resnet50(pretrained=False)
        }

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # Save original fully connected layer for feature extraction
        self.backbone.fc = nn.Identity()  # Strip FC for raw features
        self.projector = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim_mlp, out_dim)
        )

    def _get_basemodel(self, model_name):
        if model_name not in self.resnet_dict:
            raise ValueError(f"Invalid backbone: {model_name}")
        return self.resnet_dict[model_name]

    def forward(self, x):
        h = self.backbone(x)           # encoder output
        z = self.projector(F.normalize(h, dim=1))  # projection
        return z

    def get_features(self, x):
        h = self.backbone(x)
        return F.normalize(h, dim=1)


class ViTSimCLR(nn.Module):
    def __init__(self, args, image_size=32):
        super().__init__()

        hidden_size = args.vit_hidden_size
        intermediate_size = args.vit_intermediate_size or hidden_size * 4
        projection_dim = args.out_dim
        hidden_dim = args.proj_hidden_dim
        num_layers = args.proj_num_layers

        config = ViTConfig(
            image_size=image_size,
            patch_size=args.vit_patch_size,
            hidden_size=hidden_size,
            num_hidden_layers=args.vit_layers,
            num_attention_heads=args.vit_heads,
            intermediate_size=intermediate_size,
            num_channels=3,)

        self.vit = ViTModel(config)
        self.pooling = args.vit_pooling  # 'cls' or 'mean'

        

        mlp_input_dim = hidden_size

        # build projection MLP dynamically
        mlp_layers = []
        # first layer
        mlp_layers.append(nn.Linear(mlp_input_dim, hidden_dim))
        mlp_layers.append(nn.GELU())
        mlp_layers.append(nn.Dropout(p=args.dropout))
        # middle layers (if any)
        for _ in range(num_layers - 2):
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            mlp_layers.append(nn.GELU())
            mlp_layers.append(nn.Dropout(p=args.dropout))
        # final layer
        mlp_layers.append(nn.Linear(hidden_dim, projection_dim))
        mlp_layers.append(nn.LayerNorm(projection_dim))

        self.projector = nn.Sequential(*mlp_layers)


    def forward(self, x):
        features = self.get_features(x)
        z = self.projector(features)
        
        return F.normalize(z, dim=1)

    def get_features(self, x):
        out = self.vit(pixel_values=x)
        
        hidden = out.last_hidden_state
        if self.pooling == 'cls':
            features = hidden[:, 0]
        elif self.pooling == 'mean':
            features = hidden[:, 1:].mean(dim=1)
        elif self.pooling == 'both':
            cls   = hidden[:, 0]
            mean  = hidden[:, 1:].mean(dim=1)
            features = (cls + mean) * 0.5

        return features