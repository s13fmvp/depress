# 修改后的vit_1d，不需要前面的patch划分和embedding部分，从添加positional encoding开始
# modified at 20251102 用于可视化中，主要修改 flash attention输出注意力分数
import torch
from torch import nn

import math
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from flash_attn.flash_attn_interface import flash_attn_unpadded_func


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.norm = nn.LayerNorm(dim)
#         self.attend = nn.Softmax(dim = -1)
#         self.dropout = nn.Dropout(dropout)

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x):
#         x = self.norm(x)
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)

# class Attention(nn.Module):  # 带有mask版本的attention
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.norm = nn.LayerNorm(dim)

#         self.attend = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x, mask=None):  # 加入 mask 参数
#         x = self.norm(x)

#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (b, h, n, n)

#         if mask is not None:
#             mask = mask.unsqueeze(1).unsqueeze(2)  # (b, 1, 1, n)
#             mask = mask.expand(-1, self.heads, x.shape[1], -1)  # (b, h, n, n)
#             # mask shape should be (b, 1, n, n) or (b, h, n, n)
#             mask = mask.to(dtype=torch.bool)
#             dots = dots.masked_fill(~mask, float('-inf'))

#         attn = self.attend(dots)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)
    

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        bs = x.size(0)
        seq = x.size(1)

        q = qkv[0].reshape(bs, seq, self.heads, self.dim_head).transpose(1, 2)
        k = qkv[1].reshape(bs, seq, self.heads, self.dim_head).transpose(1, 2)
        v = qkv[2].reshape(bs, seq, self.heads, self.dim_head).transpose(1, 2)

        q = q / math.sqrt(self.dim_head)
        attn_logits = torch.matmul(q, k.transpose(-2, -1))  # [bs, heads, seq, seq]
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask[:, None, None, :] == 0, float('-inf'))
        attn = F.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn, v)  # [bs, heads, seq, dim_head]
        out = out.transpose(1, 2).contiguous().reshape(bs, seq, self.heads * self.dim_head)
        
        return self.to_out(out), attn  # 返回注意力矩阵用于可视化



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask)[0] + x
            x = ff(x) + x
        return x



class ViT_2d(nn.Module):
    """
    Vision Transformer for 2D inputs (e.g., images or 2D feature maps).
    Args:
        img_size: int or tuple, size of the input image (H, W)
        patch_size: int or tuple, size of each patch (ph, pw)
        num_classes: int, number of output classes
        dim: int, embedding dimension
        depth: int, number of transformer layers
        heads: int, number of attention heads
        mlp_dim: int, hidden dimension of MLP in transformer
        channels: int, number of input channels
        dim_head: int, dimension per attention head
        dropout: float, dropout rate
        emb_dropout: float, dropout rate after patch embedding
        pool: str, "cls" or "mean"
    """
    def __init__(
        self,
        img_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
        pool="cls"
    ):
        super().__init__()

        img_height = img_width = img_size
        patch_height = patch_width = patch_size


        assert img_height % patch_height == 0 and img_width % patch_width == 0, "Image dimensions must be divisible by patch size."

        num_patches = (img_height // patch_height) * (img_width // patch_width)
        height_patches = (img_height // patch_height)
        patch_dim = channels * patch_height * patch_width

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.img_height = img_height
        self.img_width = img_width
        self.num_patches = num_patches

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=patch_height, pw=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.to_img_embedding = Rearrange('b (h w) (ph pw c) -> b c (h ph) (w pw)', h=height_patches, ph=patch_height, pw=patch_width)

        self.pos_embedding = nn.Parameter(torch.randn(1, 70000 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, mask=None):
        # x: (b, c, h, w)
        b = x.shape[0]
        x = self.to_patch_embedding(x)  # (b, num_patches, dim)

        cls_tokens = self.cls_token.expand(b, -1, -1)  # (b, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (b, num_patches+1, dim)
        x = x + self.pos_embedding[:, :x.size(1)]
        x = self.dropout(x)

        x = self.transformer(x, mask=mask)

        if self.pool == "cls":
            out = x[:, 0]
        else:
            out = x[:, 1:].mean(dim=1)
        return self.mlp_head(out)



class ViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.to_img_embedding = Rearrange('b n (p c) -> b c (n p)', p = patch_size)

        self.pos_embedding = nn.Parameter(torch.randn(1, 2000, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = "cls"

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, mask=None):
        # x = self.to_patch_embedding(series)
        
        # x.shape: (batch_size, seq_len, embedding_dim)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)

        x, ps = pack([cls_tokens, x], 'b * d')

        if mask is not None:
            cls_mask = torch.ones((b, 1), dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        cls_tokens, ori_tokens = unpack(x, ps, 'b * d')

        return cls_tokens, ori_tokens



