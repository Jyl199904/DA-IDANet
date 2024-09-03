import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads   #64*8=512
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads     #8
        self.scale = dim_head ** -0.5     #1/8

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # 将输入 x 传递给 self.to_qkv，进行线性变换，将结果按照最后一个维度（-1）进行分割,得到查询（query）、键（key）和值（value）的结果。

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # 对查询、键、值进行形状转换，将维度重排，使得每个头的注意力权重可以同时计算

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # 计算查询和键的点积，并应用缩放因子，得到未归一化的注意力权重。

        attn = self.attend(dots)
        # 对注意力权重进行 softmax 归一化，得到注意力分布。
        out = torch.matmul(attn, v)
        # 将注意力权重与值相乘，得到经过注意力机制加权的值。
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 将输出进行形状重排，将头的维度合并回原始维度。
        out = self.to_out(out)
        # 对输出进行投影，如果需要的话，否则保持恒等映射。
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads   #64*8=512
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads
        # 获取输入特征 x 的形状信息，并保存在变量中，其中 b 表示批次大小，n 表示序列长度，h 表示头数。
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])
        # 对查询、键、值进行形状转换，将维度重排，使得每个头的注意力权重可以同时计算
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # 算查询和键的点积，并应用缩放因子，得到未归一化的注意力权重。
        mask_value = -torch.finfo(dots.dtype).max
        # 计算一个最小的负数，用于在没有注意力的位置进行掩码填充。
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            # 对注意力掩码进行形状转换，将其展平并进行填充，以匹配注意力权重形状。
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            # 确保掩码 mask 的最后一个维度与注意力权重 dots 的最后一个维度相匹配。
            mask = mask[:, None, :] * mask[:, :, None]
            # 计算掩码的乘积，生成合法位置为True，非法位置为False的二维掩码。
            dots.masked_fill_(~mask, mask_value)
            # 对非法位置进行填充，将注意力权重中的非法位置设置为一个非常小的负数。
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # 将注意力权重与值相乘，得到经过跨注意力机制加权的值。
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 将输出进行形状重排，将头的维度合并回原始维度。
        out = self.to_out(out)

        return out
# 通过crossattention，可以实现跨注意力机制，用于不同序列之间的信息交互和特征提取

class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio, kernel_size):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class DS_layer(nn.Module):
    def __init__(self,in_d,out_d,stride,output_padding,n_class):
        super(DS_layer,self).__init__()

        self.disconv = nn.ConvTranspose2d(in_d,out_d,kernel_size=3,padding=1,stride=stride,output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_d)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.outconv = nn.ConvTranspose2d(out_d,n_class,kernel_size=3,padding=1)

    def forward(self,input):
        x = self.disconv(input)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.outconv(x)
        return x

