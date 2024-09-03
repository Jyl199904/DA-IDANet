import torch
import torch.nn as nn



class ctb(nn.Module):
    def __init__(self, in_channels=32):
        super(ctb, self).__init__()
        self.to_key = nn.Linear(in_channels * 2, in_channels, bias=False)
        self.to_value = nn.Linear(in_channels * 2, in_channels, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.gamma_cam_lay3 = nn.Parameter(torch.zeros(1))
        self.cam_layer0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.cam_layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.cam_layer2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_feature, features):
        Query_features = input_feature
        Query_features = self.cam_layer0(Query_features)
        key_features = self.cam_layer1(features)
        value_features = self.cam_layer2(features)
        QK = torch.einsum("nlhd,nshd->nlsh", Query_features, key_features)
        softmax_temp = 1. / Query_features.size(3) ** .5
        A = torch.softmax(softmax_temp * QK, dim=2)
        queried_values = torch.einsum("nlsh,nshd->nlhd", A, value_features).contiguous()
        message = self.mlp(torch.cat([input_feature, queried_values], dim=1))
        return input_feature + message


class IDAM(nn.Module):
    def __init__(self, in_channels=32):
        super(IDAM, self).__init__()
        self.ctb = ctb(in_channels)
        self.disconv = nn.ConvTranspose2d(in_channels*2, in_channels, kernel_size=1)
        self.convs3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        def forward(self, t1, t2):
            t1 = self.ctb(t1, t2)
            t2 = self.ctb(t2, t1)
            diff = torch.abs(t1 - t2)
            t1_diff = torch.cat([t1, diff], dim=1)
            t1_diff = self.disconv(t1_diff)
            t1_diff = self.convs3(t1_diff)
            t1_final = t1 * t1_diff

            t2_diff = torch.cat([t2, diff], dim=1)
            t2_diff = self.disconv(t2_diff)
            t2_diff = self.convs3(t2_diff)
            t2_final = t2 * t2_diff

            return t1_final, t2_final
