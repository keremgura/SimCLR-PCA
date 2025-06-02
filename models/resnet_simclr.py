import torch.nn as nn
import torchvision.models as models
from transformers import ViTConfig, ViTModel
import torch.nn.functional as F
import torch

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
    def __init__(self, args, image_size=32, projection_dim=128):
        super().__init__()

        hidden_size = args.vit_hidden_size
        intermediate_size = args.vit_intermediate_size or hidden_size * 4

        config = ViTConfig(
            image_size=image_size,
            patch_size=args.vit_patch_size,
            hidden_size=hidden_size,
            num_hidden_layers=args.vit_layers,
            num_attention_heads=args.vit_heads,
            intermediate_size=intermediate_size,
            num_channels=3,
        )

        self.vit = ViTModel(config)
        self.pooling = args.vit_pooling  # 'cls' or 'mean'

        self.projector = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, projection_dim),
            nn.LayerNorm(projection_dim)
        )

    def forward(self, x):
        out = self.vit(pixel_values=x)
        if self.pooling == "cls":
            features = out.last_hidden_state[:, 0]  # CLS token
        else:
            features = out.last_hidden_state[:, 1:].mean(dim=1)  # Mean of patch tokens
            
        return self.projector(features)

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return x + self.block(x)

class PCASimCLR(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=512, dropout=0.1):
        super(PCASimCLR, self).__init__()

        # Initial linear projection to match hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Dropout(p=dropout))

        # Projection head for SimCLR
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        h = self.encoder(x)
        #z = self.projector(F.normalize(h, dim=1))
        z = F.normalize(self.projector(h), dim=1)
        return z

    def get_features(self, x):
        x = self.encoder(x)
        return F.normalize(x, dim=1)




class PCATransformerSimCLR(nn.Module):
    def __init__(self, input_dim, out_dim, patch_size=64, hidden_dim=256, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()

        # Project the full PCA vector into hidden_dim
        self.num_patches = input_dim // patch_size
        self.patch_size = patch_size

        # Patch embedding: project each patch of size patch_size to hidden_dim
        self.patch_embed = nn.Linear(patch_size, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))


        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Optional normalization after transformer
        self.norm = nn.LayerNorm(hidden_dim)

        # SimCLR projection head
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        # Input x shape: [batch_size, input_dim]
        B = x.size(0)
        if x.ndim == 4:  # [B, 3, H, W]
            x = x.view(B, -1)
        x = x.contiguous().view(B, self.num_patches, self.patch_size)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x[:, 0])                  
        return self.projector(x)

    def get_features(self, x):
        B = x.size(0)
        if x.ndim == 4:  # [B, 3, H, W]
            x = x.view(B, -1)
        x = x.contiguous().view(B, self.num_patches, self.patch_size)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x[:, 0])  
        return F.normalize(x, dim=1)



