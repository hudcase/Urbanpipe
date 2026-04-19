

import torch
import torch.nn as nn
import math


# =====================================
# Config
# =====================================
class Config:
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.):
        self.d_model = dim
        self.n_heads = heads
        self.d_k = dim_head
        self.d_v = dim_head
        self.d_ff = dim * ff_mult
        self.p_dropout = dropout


# =====================================
# GELU
# =====================================
def gelu(x):
    return 0.5 * x * (1. + torch.erf(x / math.sqrt(2.0)))


# =====================================
# Attention
# =====================================
class ScaledDotProductAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.scale = math.sqrt(config.d_k)

    def forward(self, Q, K, V):

        attn = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)

        return out


# =====================================
# MultiHeadAttention
# =====================================
class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.W_Q = nn.Linear(config.d_model,
                             config.n_heads * config.d_k,
                             bias=False)

        self.W_K = nn.Linear(config.d_model,
                             config.n_heads * config.d_k,
                             bias=False)

        self.W_V = nn.Linear(config.d_model,
                             config.n_heads * config.d_v,
                             bias=False)

        self.fc = nn.Linear(config.n_heads * config.d_v,
                            config.d_model,
                            bias=False)

        self.attention = ScaledDotProductAttention(config)

    def forward(self, x):

        B, N, D = x.shape

        h = self.config.n_heads
        d_k = self.config.d_k
        d_v = self.config.d_v

        Q = self.W_Q(x).view(B, N, h, d_k).transpose(1, 2)
        K = self.W_K(x).view(B, N, h, d_k).transpose(1, 2)
        V = self.W_V(x).view(B, N, h, d_v).transpose(1, 2)

        context = self.attention(Q, K, V)

        context = context.transpose(1, 2).contiguous().view(B, N, h*d_v)

        out = self.fc(context)

        return out


# =====================================
# FeedForward
# =====================================
class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)

        self.dropout = nn.Dropout(config.p_dropout)

    def forward(self, x):

        x = self.fc1(x)
        x = gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# =====================================
# TimeSformer Block (Divided Attention)
# =====================================
class TimeSformerBlock(nn.Module):

    def __init__(self, config, num_frames, num_patches):
        super().__init__()

        self.num_frames = num_frames
        self.num_patches = num_patches

        # Spatial attention
        self.spatial_norm = nn.LayerNorm(config.d_model)
        self.spatial_attn = MultiHeadAttention(config)

        # Temporal attention
        self.temporal_norm = nn.LayerNorm(config.d_model)
        self.temporal_attn = MultiHeadAttention(config)

        # FFN
        self.ff_norm = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(self, x):
        """
        x shape:
        (B, 1 + F*P, D)
        """

        B, N, D = x.shape

        cls = x[:, 0:1]     # (B,1,D)
        tokens = x[:, 1:]   # (B,F*P,D)

        F = self.num_frames
        P = self.num_patches

        # reshape
        tokens = tokens.view(B, F, P, D)

        # =====================================
        # 1. Spatial Attention (论文第一步)
        # 每一帧内部做 attention
        # =====================================

        spat = tokens.reshape(B*F, P, D)

        spat = spat + self.spatial_attn(
            self.spatial_norm(spat)
        )

        tokens = spat.view(B, F, P, D)

        # =====================================
        # 2. Temporal Attention (论文第二步)
        # 每个patch位置 across frame attention
        # =====================================

        temp = tokens.permute(0, 2, 1, 3)  # (B,P,F,D)

        temp = temp.reshape(B*P, F, D)

        temp = temp + self.temporal_attn(
            self.temporal_norm(temp)
        )

        tokens = temp.view(B, P, F, D).permute(0, 2, 1, 3)

        # =====================================
        # merge back
        # =====================================

        tokens = tokens.reshape(B, F*P, D)

        x = torch.cat([cls, tokens], dim=1)

        # =====================================
        # FFN
        # =====================================

        x = x + self.ff(
            self.ff_norm(x)
        )

        return x


# =====================================
# TimeSformer
# =====================================
class TimeSformer(nn.Module):

    def __init__(
        self,
        dim=512,
        image_size=224,
        patch_size=16,
        num_frames=8,
        num_classes=10,
        depth=4,
        heads=8,
        dim_head=64
    ):
        super().__init__()

        assert image_size % patch_size == 0

        self.patch_size = patch_size
        self.num_frames = num_frames

        patches_per_frame = (image_size//patch_size)**2

        patch_dim = 3*patch_size*patch_size

        self.patch_embed = nn.Linear(patch_dim, dim)

        self.cls_token = nn.Parameter(torch.randn(1,1,dim))

        self.pos = nn.Parameter(
            torch.randn(1, 1 + num_frames*patches_per_frame, dim)
        )

        config = Config(dim, heads, dim_head)

        self.blocks = nn.ModuleList([
            TimeSformerBlock(config, num_frames, patches_per_frame)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

        self.head = nn.Linear(dim, num_classes)


    def forward(self, video):

        B,F,C,H,W = video.shape

        p = self.patch_size

        h = H//p
        w = W//p

        P = h*w

        video = video.reshape(B,F,C,h,p,w,p)
        video = video.permute(0,1,3,5,4,6,2)
        video = video.reshape(B,F*P,p*p*C)

        x = self.patch_embed(video)

        cls = self.cls_token.expand(B,-1,-1)

        x = torch.cat([cls,x],dim=1)

        x = x + self.pos[:,:x.shape[1]]

        for block in self.blocks:
            x = block(x)

        cls = x[:,0]

        out = self.head(self.norm(cls))

        return out


# =====================================
# Test
# =====================================
if __name__ == "__main__":

    model = TimeSformer()

    video = torch.randn(2,8,3,224,224)

    out = model(video)

    print(video.shape)
    print(out.shape)
