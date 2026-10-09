import torch
import torch.nn as nn
import numpy as np
import math
import einops
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import sys
import torch.nn.functional as F
from preprocess.utils import pca_with_torch

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SelfAttention2(nn.Module):
    def __init__(self, feature_dim, hidden_size):
        super(SelfAttention2, self).__init__()
        self.key = nn.Linear(feature_dim, feature_dim)
        self.query = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.out = nn.Linear(feature_dim, hidden_size)

    def forward(self, x):
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(K.size(-1), dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        output = attention_output.mean(dim=1)

        return self.out(output)
class Attention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        C, G = x.shape
        qkv = self.qkv(x).reshape(C, 3, self.num_heads, G // self.num_heads)
        qkv = einops.rearrange(qkv, 'c n h fph -> n h c fph')
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 1)
        x = einops.rearrange(x, 'c h fph -> c (h fph)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k, v):
        C, G = x.shape
        qkv = torch.concat((self.q_proj(x), self.k_proj(k), self.v_proj(v)), dim=-1).reshape(C, 3, self.num_heads,
                                                                                             G // self.num_heads).permute(
            1, 2, 0, 3)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(C, G)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def modulate(x, shift, scale):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    res = x * (1 + scale) + shift
    return res


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class CrossDiTblock(nn.Module):
    def __init__(self,
                 feature_dim=2000,
                 mlp_ratio=4.0,
                 num_heads=10,
                 **kwargs) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(num_features=feature_dim, elementwise_affine=False, eps=1e-4)
        self.attn = Attention2(feature_dim, num_heads=num_heads, qkv_bias=True, **kwargs)

        self.norm2 = nn.LayerNorm(num_features=feature_dim, elementwise_affine=False, eps=1e-4)
        self.cross_attn = CrossAttention(feature_dim, num_heads=num_heads, qkv_bias=True, **kwargs)

        self.norm3 = nn.LayerNorm(num_features=feature_dim, elementwise_affine=False, eps=1e-4)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        mlp_hidden_dim = int(feature_dim * mlp_ratio)
        self.mlp = Mlp(in_features=feature_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, c):

        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), c, c)
        x = x + self.mlp(self.norm3(x))
        return x


class DiTblock(nn.Module):
    def __init__(self,
                 feature_dim=2000,
                 mlp_ratio=4.0,
                 num_heads=10,
                 **kwargs) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-4)
        self.attn = Attention2(feature_dim, num_heads=num_heads, qkv_bias=True, **kwargs)

        self.norm2 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-4)
        approx_gelu = lambda: nn.GELU()

        mlp_hidden_dim = int(feature_dim * mlp_ratio)
        self.mlp = Mlp(in_features=feature_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feature_dim, 6 * feature_dim, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):

    def __init__(self, hidden_size, out_size):
        super().__init__()

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-4)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


BaseBlock = {'dit': DiTblock,
             'cross_dit': CrossDiTblock}

class UNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )
        self.middle = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_features)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3


class DiT_diff(nn.Module):
    def __init__(self,
                 st_input_size,
                 condi_input_size,
                 hidden_size,
                 depth,
                 dit_type,
                 num_heads,
                 classes,
                 pca_dim,
                 gene_num,
                 mlp_ratio=4.0,
                 **kwargs) -> None:
        super().__init__()

        self.st_input_size = st_input_size
        self.condi_input_size = condi_input_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.classes = classes
        self.mlp_ratio = mlp_ratio
        self.dit_type = dit_type
        self.pca_dim = pca_dim
        self.gene_num = gene_num
        self.in_layer = nn.Sequential(
            nn.Linear(st_input_size, hidden_size),
        )
        self.x_in_layer = nn.Sequential(
            nn.Linear(condi_input_size, hidden_size)
        )
        self.cond_layer = nn.Sequential(
            nn.Linear(self.condi_input_size, hidden_size),
        )

        self.cond_layer_atten= SelfAttention2(self.condi_input_size, self.hidden_size)
        self.cond_layer_mlp = SimpleMLP(self.gene_num, self.hidden_size, self.hidden_size*2)
        self.condi_emb = nn.Embedding(classes, hidden_size)
        self.unet = UNet(in_features=hidden_size * 2, out_features=self.st_input_size)
        self.time_emb = TimestepEmbedder(hidden_size=self.hidden_size *2)


        self.blks = nn.ModuleList([
            BaseBlock[dit_type](self.hidden_size * 2, mlp_ratio=self.mlp_ratio, num_heads=self.num_heads) for _ in
            range(self.depth)
        ])

        self.out_layer = FinalLayer(self.hidden_size*2, self.st_input_size)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.condi_emb.weight, std=0.02)

        nn.init.normal_(self.time_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_emb.mlp[2].weight, std=0.02)

        if self.dit_type == 'dit':
            for block in self.blks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.out_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.out_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_layer.linear.weight, 0)
        nn.init.constant_(self.out_layer.linear.bias, 0)

    def forward(self, x, x_hat, t, y, **kwargs):
        x = x.float()
        x_hat = x_hat.float()
        batch_size = x.shape[0]

        x_hat = self.x_in_layer(x_hat)

        t = self.time_emb(t)

        y_mean = y.mean(dim=0)
        y_expanded = y_mean.unsqueeze(0).expand(batch_size, -1)
        x_cond = self.cond_layer_mlp(y_expanded)

        c = t + x_cond

        x = self.in_layer(x)
        x = torch.cat([x, x_hat], dim=1)

        for blk in self.blks:
            x = blk(x, c)
        x = self.out_layer(x, c)
        return x
