import torch
import torch.nn.functional as F
from torch import nn

class cross_entropy(nn.Module):
    def __init__(self, weight=None, reduction='mean',ignore_index=256):
        super(cross_entropy, self).__init__()
        self.weight = weight
        self.ignore_index =ignore_index
        self.reduction = reduction


    def forward(self,input, target):
        target = target.long()
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)
        if input.shape[-1] != target.shape[-1]:
            input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

        return F.cross_entropy(input=input, target=target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


def compute_mmd(source_features, target_features, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
    def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1), total.size(2), total.size(3))
        total1 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1), total.size(2), total.size(3))

        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    batch_size = source_features.size(0)
    kernels = guassian_kernel(source_features, target_features, kernel_mul=kernel_mul, kernel_num=kernel_num)

    XX = torch.mean(kernels[:batch_size, :batch_size])
    YY = torch.mean(kernels[batch_size:, batch_size:])
    XY = torch.mean(kernels[:batch_size, batch_size:])
    YX = torch.mean(kernels[batch_size:, :batch_size])

    loss = XX + YY - XY - YX

    return loss