# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch
import torch.nn as nn

from utils import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob   # probability of keeping a path     
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    
    # x.shape  = batch size
    # shape = (batch size, 1, 1, 1) if x.ndim = 4
    
    #x.ndim = 3 (1,) * (x.ndim - 1) → (1, 1, 1) # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()

        # Number of attention heads
        self.num_heads = num_heads

        # Dimension per head
        # dim = num_heads * head_dim
        head_dim = dim // num_heads

        # Scaling factor for dot-product attention
        self.scale = qk_scale or head_dim ** -0.5

        # Linear layer to produce Q, K, V all at once
        # Input:  (B, N, C)
        # Output: (B, N, 3C)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        # Final projection back to original feature dimension
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x shape: (B, N, C)
        # B = batch size
        # N = number of tokens (patches)
        # C = embedding dimension
        B, N, C = x.shape

        # Apply linear projection
        # (B, N, C) -> (B, N, 3C)
        qkv = self.qkv(x)

        # Reshape to separate Q, K, V and heads
        # (B, N, 3C) ->
        # (B, N, 3, num_heads, head_dim)
        qkv = qkv.reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        )

        # Permute to put QKV first
        # (3, B, num_heads, N, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Split into query, key, value
        # Each: (B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        # k.transpose(-2, -1): (B, num_heads, head_dim, N)
        # q @ k^T -> (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Softmax over last dimension (keys)
        attn = attn.softmax(dim=-1)

        # Dropout on attention weights
        attn = self.attn_drop(attn)

        # Weighted sum of values
        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim)
        # -> (B, num_heads, N, head_dim)
        x = attn @ v

        # Reorder and merge heads
        # (B, num_heads, N, head_dim)
        # -> (B, N, num_heads, head_dim)
        # -> (B, N, C)
        x = x.transpose(1, 2).reshape(B, N, C)

        # Final linear projection
        x = self.proj(x)

        # Output dropout
        x = self.proj_drop(x)

        # Return:
        # x:    (B, N, C)
        # attn: (B, num_heads, N, N)
        return x, attn



class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    first it divides the image into patches, (8,1,224,224) 8 image, 1 channel, 224 height, 224 width
    it projects each patch to 768 and we'll get the size of (8,196,768) 8 image, 196 patches, 768 embedding dimension
    224/16=14, 14*14=196
    """
    # convert the image into patches 
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768): 
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size) # total number of patches in height and width
        self.img_size = img_size # size of the image
        self.patch_size = patch_size # size of each patch
        self.num_patches = num_patches  # total number of patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)  # project the patches into embedding space

    def forward(self, x):
        
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=1, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed( # output size: (B, num_patches, embed_dim)
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # shape of cls token: (1, 1, embed_dim) 0000
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        """
        Prepares input tokens for the Vision Transformer.

        Steps:
        1. Convert images into patch embeddings
        2. Prepend the CLS token
        3. Add positional embeddings
        4. Apply dropout for regularization

        Input:
            x: Tensor of shape (B, C_in, H, W)
            B = batch size
            C_in = input channels (1 for grayscale, 3 for RGB)
            H, W = image height and width
        Output:
            tokens: Tensor of shape (B, N+1, embed_dim)
                    N = number of patch tokens
                    +1 = CLS token
        """

        # Extract batch size and image dimensions
        B, nc, w, h = x.shape
        # B = batch size
        # nc = input channels
        # w, h = width and height of the input image

        # Convert image to patch embeddings
        # PatchEmbed splits image into patches, flattens each patch, and projects it to embed_dim
        # Input:  x -> (B, nc, w, h)
        # Output: x -> (B, N, embed_dim), where N = number of patches
        x = self.patch_embed(x)

        # Create CLS token for each batch element
        # self.cls_token has shape (1, 1, embed_dim)
        # We expand it to match batch size B
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # cls_tokens shape -> (B, 1, embed_dim)

        # Prepend CLS token to patch embeddings
        # Concatenate along the token (sequence) dimension
        x = torch.cat((cls_tokens, x), dim=1)
        # x shape -> (B, N+1, embed_dim)

        # Add positional embeddings
        # Each token receives a positional encoding to retain spatial information
        # interpolate_pos_encoding adjusts positional embeddings for image size if needed
        x = x + self.interpolate_pos_encoding(x, w, h) # patch embeddings + pos embeddings      
        # x shape remains -> (B, N+1, embed_dim)

        # Apply dropout for regularization
        # Dropout randomly zeroes some token embeddings to prevent overfitting
        x = self.pos_drop(x)
        # final output shape -> (B, N+1, embed_dim)

        return x

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output












def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x



# the test of the full model
if __name__ == "__main__":
    backbone = vit_base(patch_size=16, img_size=[128])
    DINO = DINOHead(in_dim=backbone.embed_dim, out_dim=65536, nlayers=3)
    img = torch.randn(1, 1, 128, 128)
    feat = backbone(img)
    print("feat shape:", feat.shape)
    out = DINO(feat)
    print("out shape:", out.shape)