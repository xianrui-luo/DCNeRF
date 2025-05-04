import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from H_align.core.loss import batch_episym

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1]  # (batch_size, num_points, k)

    return idx[:, :, 1:]


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)  # (batch_size, num_points, k)
    else:
        idx_out = idx
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx_out + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


#    feature = feature.permute(0, 3, 1, 2).contiguous()

#    return feature

class ACN(nn.Module):
    def __init__(self, num_groups, channel):
        super(ACN, self).__init__()
        self.conv = nn.Conv2d(channel, 1, (1, 1))
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=channel)

    #    self.gn = nn.BatchNorm2d(channel)
    def normalization(self, x, w):
        mean = torch.sum(x * w, dim=2, keepdim=True)
        std = torch.sum(((x - mean) ** 2) * w, dim=2, keepdim=True)
        std = std ** (0.5)
        return (x - mean) / (std + 1e-10)

    def forward(self, x):
        w_local = self.conv(x
        )
        w = F.sigmoid(w_local)
        w = F.softmax(w, dim=2)
        w = w / (torch.norm(w, p=1, dim=2, keepdim=True) + 1e-10)
        out = self.normalization(x, w)
        out = self.gn(out)
        return out, w_local.view(w_local.size(0), -1)


def check_D(D, label):
    in_r = D.squeeze(1) * label
    in_i = torch.nonzero(in_r[0])
    print("Inlier_d:", in_r[0][in_i])
    in_r = D.squeeze(1) * (1 - label)
    in_i = torch.nonzero(in_r[0])
    print("Outlier_d:", in_r[0][in_i])
    exit()


class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        # self.left_conv1 = nn.Conv2d(inchannel, outchannel, (1, 1))
        self.left_conv1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )
        self.left_ac1 = ACN(8, outchannel)
        # self.left_conv2 = nn.Conv2d(outchannel, outchannel, (1, 1))
        self.left_conv2 = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )
        self.left_ac2 = ACN(8, outchannel)

    def forward(self, x):
        w = []
        x1 = self.right(x) if self.pre is True else x
        out = x + self.left_conv1(x)
        out, w1 = self.left_ac1(out)
        out = F.relu(out)
        out = out + self.left_conv2(out)
        out, w2 = self.left_ac2(out)
        out = out + x1

        w.append(w1)
        w.append(w2)
        return F.relu(out), w


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.symeig(X[batch_idx, :, :].squeeze(), True)
        bv[batch_idx, :, :] = v
    bv = bv.to(device)

    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)

    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


class NM_block_v2_1(nn.Module):
    def __init__(self, inchannel):
        super(NM_block_v2_1, self).__init__()
        self.inchannel = inchannel

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.inchannel, 128, (1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, (1, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 128, (1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Conv2d(128, 1, (1, 1))

        self.res1 = ResNet_Block(128, 128, pre=False)
        self.res2 = ResNet_Block(128, 128, pre=False)
        self.res3 = ResNet_Block(128, 128, pre=False)
        self.res4 = ResNet_Block(128, 128, pre=False)

    def self_attention(self, x):
        out = self.conv1(x)

        out = out.squeeze(-1)
        out = out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-5)
        A = torch.bmm(out.permute(0, 2, 1).contiguous(), out)

        eye = torch.eye(A.size(1)).unsqueeze(0).repeat(A.size(0), 1, 1).to(device)
        D = torch.sum(A - eye, dim=1, keepdim=True)
        out = torch.cat([x, torch.relu(torch.tanh(D.unsqueeze(-1)))], dim=1)

        return out, D.view(D.size(0), -1)

    def forward(self, x):
        out = x.transpose(1, 3).contiguous()
        out, D = self.self_attention(out)

        idx = knn(out.squeeze(-1).contiguous(), k=8)
        out = get_graph_feature(out, k=8, idx=idx)
        out = self.conv2(out)
        out = F.max_pool2d(out, (1, 8))

        out, w1 = self.res1(out)
        out, w2 = self.res2(out)
        out, w3 = self.res3(out)
        out, w4 = self.res4(out)

        logit = self.linear(out)
        logit = logit.view(logit.size(0), -1)
        w = [D] + [logit]
        e_hat = weighted_8points(x, w[-1])
        return w, e_hat


class NM_block_v2_2(nn.Module):
    def __init__(self, inchannel, k_n):
        super(NM_block_v2_2, self).__init__()
        self.inchannel = inchannel
        self.k_n = k_n

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.inchannel, 128, (1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, (1, 1))
        )

        self.conv = nn.Sequential(
            nn.Conv2d(self.inchannel * 2 + 2, 128, (1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResNet_Block(128, 128, pre=False)
        self.res2 = ResNet_Block(128, 128, pre=False)
        self.res3 = ResNet_Block(128, 128, pre=False)
        self.res4 = ResNet_Block(128, 128, pre=False)

        self.linear = nn.Conv2d(128, 1, (1, 1))

    def self_attention(self, x):
        out = self.conv1(x)

        out = out.squeeze(-1)
        out = out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-5)
        A = torch.bmm(out.permute(0, 2, 1).contiguous(), out)

        eye = torch.eye(A.size(1)).unsqueeze(0).repeat(A.size(0), 1, 1).to(device)
        D = torch.sum(A - eye, dim=1, keepdim=True)
        out = torch.cat([x, torch.relu(torch.tanh(D.unsqueeze(-1)))], dim=1)

        return out, D.view(D.size(0), -1)

    def forward(self, data, x):
        out = data.transpose(1, 3).contiguous()
        out, D = self.self_attention(out)

        idx = knn(out.squeeze(-1).contiguous(), k=self.k_n)
        out = get_graph_feature(out, k=self.k_n, idx=idx)
        out = self.conv(out)
        out = F.max_pool2d(out, (1, self.k_n))
        out, w1 = self.res1(out)
        out, w2 = self.res2(out)
        out, w3 = self.res3(out)
        out, w4 = self.res4(out)

        logit = self.linear(out)
        logit = logit.view(logit.size(0), -1)
        w = [D] + [logit]

        e_hat = weighted_8points(x, w[-1])
        return w, e_hat


class NM_Net_v2(nn.Module):
    def __init__(self, config):
        super(NM_Net_v2, self).__init__()
        self.obj_geod_th = config.obj_geod_th
        #    self.block1 = NM_block(4, config, initial=True)
        #    self.block2 = NM_block(6, config, initial=False)
        self.block1 = NM_block_v2_1(4)
        self.block2 = NM_block_v2_2(6, k_n=8)
        self.block3 = NM_block_v2_2(6, k_n=8)
        #    self.block4 = NM_block_v2_2(12, k_n=4)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = x.unsqueeze(1)

        w1, e_hat1 = self.block1(x)
        out1 = torch.stack(w1).permute(1, 2, 0).contiguous()
        out1 = out1.unsqueeze(1)
        x_ = torch.cat([x, torch.relu(torch.tanh(out1)).detach()], dim=-1)

        #    residual = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat1).reshape(x.size(0), 1, x.size(2), 1)
        #    x_ = torch.cat([x, torch.relu(torch.tanh(out1)).detach(), residual.detach()], dim=-1)

        w2, e_hat2 = self.block2(x_, x)
        out2 = torch.stack(w2).permute(1, 2, 0).contiguous()
        out2 = out2.unsqueeze(1)
        x_ = torch.cat([x, torch.relu(torch.tanh(out2)).detach()], dim=-1)
        #    residual = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat2).reshape(x.size(0), 1, x.size(2), 1)
        #    x_ = torch.cat([x, torch.relu(torch.tanh(out2)).detach(), residual.detach()], dim=-1)

        w3, e_hat3 = self.block3(x_, x)
        A = x.new_ones(1)
        in_ratio = x.new_ones(1)

        return [w1, w2, w3], A, in_ratio, [e_hat1, e_hat2, e_hat3], [x, x, x]
