import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import build_backbone
from .modules import TransformerDecoder, Transformer
from einops import rearrange
from loss.losses import compute_mmd
from .cross_transformer import IDAM


class token_encoder(nn.Module):
    def __init__(self, in_chan = 32, token_len = 4, heads = 8):
        super(token_encoder, self).__init__()
        self.token_len = token_len
        self.conv_a = nn.Conv2d(in_chan, token_len, kernel_size=1, padding=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, token_len, in_chan))

        # 位置编码的目的是为了在处理特征图时，引入位置信息，让模型能够区分不同位置的特征。
        # 在 Transformer 等模型中，位置编码常常被用于加入空间信息，以帮助模型捕捉局部和全局的关系。
        self.transformer = Transformer(dim=in_chan, depth=1, heads=heads, dim_head=64, mlp_dim=64, dropout=0)

    def forward(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()

        tokens = torch.einsum('bln, bcn->blc', spatial_attention, x)
        # 将空间注意力权重 spatial_attention 与输入特征图 x 进行矩阵乘法操作，得到表示令牌的 tokens。矩阵乘法的结果是对输入特征图 x 中的每个像素进行了空间注意力加权，从而得到了对应的令牌表示。
        # 这个操作可以帮助模型在特定位置对特征进行更加集中的处理，并捕捉到不同位置的关系。
        tokens += self.pos_embedding
        x = self.transformer(tokens)
        return x

class token_decoder(nn.Module):
    def __init__(self, in_chan = 32, size = 32, heads = 8):
        super(token_decoder, self).__init__()
        self.pos_embedding_decoder = nn.Parameter(torch.randn(1, in_chan, size, size))
        # print("pos_embedding_decoder shape:", self.pos_embedding_decoder.shape)
        self.transformer_decoder = TransformerDecoder(dim=in_chan, depth=1, heads=heads, dim_head=True, mlp_dim=in_chan*2, dropout=0,softmax=in_chan)

    def forward(self, x, m):
        b, c, h, w = x.shape
        # pos_embedding = self.pos_embedding_decoder.expand(b, -1, -1, -1)
        # print("pos_embedding shape:", pos_embedding.shape)
        # x = self.pos_embedding_decoder.view(b,-1,-1,-1)
        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        # 将重排后的令牌特征图 x 和编码后的原始特征图 m 传递给解码器 transformer_decoder 进行特征聚合和解码。
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x


class context_aggregator(nn.Module):
    def __init__(self, in_chan=32, size=32):
        super(context_aggregator, self).__init__()
        self.token_encoder = token_encoder(in_chan=in_chan, token_len=4)
        self.token_decoder = token_decoder(in_chan = 32, size = size, heads = 8)

    def forward(self, feature):
        token = self.token_encoder(feature)
        out = self.token_decoder(feature, token)
        return out

class Classifier(nn.Module):
    def __init__(self, in_chan=32, n_class=2):
        super(Classifier, self).__init__()
        self.head = nn.Sequential(
                            nn.Conv2d(in_chan * 2, in_chan, kernel_size=3, padding=1, stride=1, bias=False),
                            nn.BatchNorm2d(in_chan),
                            nn.ReLU(),
                            nn.Conv2d(in_chan, n_class, kernel_size=3, padding=1, stride=1))
    def forward(self, x):
        x = self.head(x)
        return x


class CDNet(nn.Module):
    def __init__(self,  backbone='resnet18', output_stride=16, img_size = 256, img_chan=3, chan_num = 32, n_class =2, ratio = 8, kernel=7):
        super(CDNet, self).__init__()
        BatchNorm = nn.BatchNorm2d
        self.idam = IDAM(in_channels=32)
        self.backbone = build_backbone(backbone, output_stride, BatchNorm, img_chan)

        self.CA_s8 = context_aggregator(in_chan=chan_num, size=img_size//8)
        self.CA_s4 = context_aggregator(in_chan=chan_num, size=img_size//4)
        self.CA_s2 = context_aggregator(in_chan=chan_num, size=img_size//2)

        self.conv_s4 = nn.Conv2d(chan_num*2, chan_num, kernel_size=3, padding=1)
        self.conv_s2 = nn.Conv2d(chan_num*2, chan_num, kernel_size=3, padding=1)

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)

        self.classifier = Classifier(n_class = n_class)

    def forward(self, img1, img2):
        # CNN backbone, feature extractor
        out1_s8, out1_s4, out1_s2 = self.backbone(img1)       #(1,32,32,32), (1,32,64,64), (1,32,128,128)
        out2_s8, out2_s4, out2_s2 = self.backbone(img2)

        # context aggregate (scale 16, scale 8, scale 4)
        x1_s8= self.CA_s8(out1_s8)  #(1,32,32,32)
        x2_s8 = self.CA_s8(out2_s8)
        mmd_loss1 = compute_mmd(out1_s8,out2_s8)

        x1_s8_final,x2_s8_final = self.idam(x1_s8,x1_s8)

        x8 = torch.cat([x1_s8_final, x2_s8_final], dim=1) #(1,64,32,32)
        x8 = F.interpolate(x8, size=img1.shape[2:], mode='bicubic', align_corners=True)   #(1,64,256,256)
        x8 = self.classifier(x8)   #(1,2,256,256)

        out1_s4 = self.conv_s4(torch.cat([self.upsamplex2(x1_s8), out1_s4], dim=1))  #(1,32,64,64)
        out2_s4 = self.conv_s4(torch.cat([self.upsamplex2(x2_s8), out2_s4], dim=1))

        x1_s4 = self.CA_s4(out1_s4)  #(1,32,64,64)
        x2_s4 = self.CA_s4(out2_s4)

        mmd_loss2 = compute_mmd(out1_s4, out2_s4)
        x1_s4_final,x2_s4_final = self.idam(x1_s4,x2_s4)
        x4 = torch.cat([x1_s4_final, x2_s4_final], dim=1) #(1,64,64,64)
        x4 = F.interpolate(x4, size=img1.shape[2:], mode='bicubic', align_corners=True) #(1,64,256,256)
        x4 = self.classifier(x4) #(1,2,256,256)

        out1_s2 = self.conv_s2(torch.cat([self.upsamplex2(x1_s4), out1_s2], dim=1))  #(1,32,128,128)
        out2_s2 = self.conv_s2(torch.cat([self.upsamplex2(x2_s4), out2_s2], dim=1))

        x1 = self.CA_s2(out1_s2)  # (1,32,128,128)
        x2 = self.CA_s2(out2_s2)
        mmd_loss3 = compute_mmd(out1_s2, out2_s2)

        x1 = self.ctf(x1, x2)
        x2= self.ctf(x2, x1)
        x1_final,x2_final = self.idam(x1,x2)

        x = torch.cat([x1_final, x2_final], dim=1)    #(1,64,64,64)
        x = F.interpolate(x, size=img1.shape[2:], mode='bicubic', align_corners=True)  #(1,64,256,256)
        x = self.classifier(x)  #(1,2,256,256)
        return x,  x4, x8, mmd_loss1, mmd_loss2, mmd_loss3

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
