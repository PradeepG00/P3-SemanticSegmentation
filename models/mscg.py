import torch
import torch.nn as nn
import torch.nn.functional as F
from pretrainedmodels import se_resnext50_32x4d, se_resnext101_32x4d


def load_model(name='MSCG-Rx50', classes=7, node_size=(32, 32)):
    if name == 'MSCG-Rx50':
        net = RX50GCN3Head4Channel(out_channels=classes)
    elif name == 'MSCG-Rx101':
        net = RX101GCN3Head4Channel(out_channels=classes)
    else:
        print('not found the models')
        return -1

    return net


class RX50GCN3Head4Channel(nn.Module):
    def __init__(self, out_channels=7, pretrained=True,
                 nodes=(32, 32), dropout=0,
                 enhance_diag=True, aux_pred=True):
        super(RX50GCN3Head4Channel, self).__init__()  # same with  res_fdcs_v5

        self.aux_pred = aux_pred
        self.node_size = nodes
        self.num_cluster = out_channels

        resnet = se_resnext50_32x4d()
        self.layer0, self.layer1, self.layer2, self.layer3, = \
            resnet.layer0, resnet.layer1, resnet.layer2, resnet.layer3

        self.conv0 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        for child in self.layer0.children():
            for param in child.parameters():
                par = param
                break
            break

        self.conv0.parameters = torch.cat([par[:, 0, :, :].unsqueeze(1), par], 1)
        self.layer0 = torch.nn.Sequential(self.conv0, *list(self.layer0)[1:4])

        self.graph_layers1 = GCN_Layer(1024, 128, bnorm=True, activation=nn.ReLU(True), dropout=dropout)

        self.graph_layers2 = GCN_Layer(128, out_channels, bnorm=False, activation=None)

        self.scg = SCGBlock(in_ch=1024,
                            hidden_ch=out_channels,
                            node_size=nodes,
                            add_diag=enhance_diag,
                            dropout=dropout)

        weight_xavier_init(self.graph_layers1, self.graph_layers2, self.scg)

    def forward(self, x):
        x_size = x.size()

        gx = self.layer3(self.layer2(self.layer1(self.layer0(x))))
        gx90 = gx.permute(0, 1, 3, 2)
        gx180 = gx.flip(3)
        B, C, H, W = gx.size()

        A, gx, loss, z_hat = self.scg(gx)
        gx, _ = self.graph_layers2(
            self.graph_layers1((gx.reshape(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx += z_hat
        gx = gx.reshape(B, self.num_cluster, self.node_size[0], self.node_size[1])

        A, gx90, loss2, z_hat = self.scg(gx90)
        gx90, _ = self.graph_layers2(
            self.graph_layers1((gx90.reshape(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx90 += z_hat
        gx90 = gx90.reshape(B, self.num_cluster, self.node_size[1], self.node_size[0])
        gx90 = gx90.permute(0, 1, 3, 2)
        gx += gx90

        A, gx180, loss3, z_hat = self.scg(gx180)
        gx180, _ = self.graph_layers2(
            self.graph_layers1((gx180.reshape(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx180 += z_hat
        gx180 = gx180.reshape(B, self.num_cluster, self.node_size[0], self.node_size[1])
        gx180 = gx180.flip(3)
        gx += gx180

        gx = F.interpolate(gx, (H, W), mode='bilinear', align_corners=False)

        if self.training:
            return F.interpolate(gx, x_size[2:], mode='bilinear', align_corners=False), loss + loss2 + loss3
        else:
            return F.interpolate(gx, x_size[2:], mode='bilinear', align_corners=False)


class RX101GCN3Head4Channel(nn.Module):
    def __init__(self, out_channels=7, pretrained=True,
                 nodes=(32, 32), dropout=0,
                 enhance_diag=True, aux_pred=True):
        super(RX101GCN3Head4Channel, self).__init__()  # same with  res_fdcs_v5

        self.aux_pred = aux_pred
        self.node_size = nodes
        self.num_cluster = out_channels

        resnet = se_resnext101_32x4d()
        self.layer0, self.layer1, self.layer2, self.layer3, = \
            resnet.layer0, resnet.layer1, resnet.layer2, resnet.layer3

        self.conv0 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        for child in self.layer0.children():
            for param in child.parameters():
                par = param
                break
            break

        self.conv0.parameters = torch.cat([par[:, 0, :, :].unsqueeze(1), par], 1)
        self.layer0 = torch.nn.Sequential(self.conv0, *list(self.layer0)[1:4])

        self.graph_layers1 = GCN_Layer(1024, 128, bnorm=True, activation=nn.ReLU(True), dropout=dropout)

        self.graph_layers2 = GCN_Layer(128, out_channels, bnorm=False, activation=None)

        self.scg = SCGBlock(in_ch=1024,
                            hidden_ch=out_channels,
                            node_size=nodes,
                            add_diag=enhance_diag,
                            dropout=dropout)

        weight_xavier_init(self.graph_layers1, self.graph_layers2, self.scg)

    def forward(self, x):
        x_size = x.size()

        gx = self.layer3(self.layer2(self.layer1(self.layer0(x))))
        gx90 = gx.permute(0, 1, 3, 2)
        gx180 = gx.flip(3)

        B, C, H, W = gx.size()

        A, gx, loss, z_hat = self.scg(gx)

        gx, _ = self.graph_layers2(
            self.graph_layers1((gx.view(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx += z_hat
        gx = gx.view(B, self.num_cluster, self.node_size[0], self.node_size[1])

        A, gx90, loss2, z_hat = self.scg(gx90)
        gx90, _ = self.graph_layers2(
            self.graph_layers1((gx90.view(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx90 += z_hat
        gx90 = gx90.view(B, self.num_cluster, self.node_size[1], self.node_size[0])
        gx90 = gx90.permute(0, 1, 3, 2)
        gx += gx90

        A, gx180, loss3, z_hat = self.scg(gx180)
        gx180, _ = self.graph_layers2(
            self.graph_layers1((gx180.view(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx180 += z_hat
        gx180 = gx180.view(B, self.num_cluster, self.node_size[0], self.node_size[1])
        gx180 = gx180.flip(3)
        gx += gx180

        gx = F.interpolate(gx, (H, W), mode='bilinear', align_corners=False)

        if self.training:
            return F.interpolate(gx, x_size[2:], mode='bilinear', align_corners=False), loss + loss2 + loss3
        else:
            return F.interpolate(gx, x_size[2:], mode='bilinear', align_corners=False)


class SCGBlock(nn.Module):
    def __init__(self, in_ch, hidden_ch=6, node_size=(32, 32), add_diag=True, dropout=0.2):
        super(SCGBlock, self).__init__()
        self.node_size = node_size
        self.hidden = hidden_ch
        self.nodes = node_size[0] * node_size[1]
        self.add_diag = add_diag
        self.pool = nn.AdaptiveAvgPool2d(node_size)

        self.mu = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=True),
            nn.Dropout(dropout),
        )

        self.logvar = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 1, 1, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, C, H, W = x.size()
        gx = self.pool(x)

        mu, log_var = self.mu(gx), self.logvar(gx)

        if self.training:
            std = torch.exp(log_var.reshape(B, self.nodes, self.hidden))
            eps = torch.randn_like(std)
            z = mu.reshape(B, self.nodes, self.hidden) + std * eps
        else:
            z = mu.reshape(B, self.nodes, self.hidden)

        A = torch.matmul(z, z.permute(0, 2, 1))
        A = torch.relu(A)

        Ad = torch.diagonal(A, dim1=1, dim2=2)
        mean = torch.mean(Ad, dim=1)
        gama = torch.sqrt(1 + 1.0 / mean).unsqueeze(-1).unsqueeze(-1)

        dl_loss = gama.mean() * torch.log(Ad[Ad < 1] + 1.e-7).sum() / (A.size(0) * A.size(1) * A.size(2))

        kl_loss = -0.5 / self.nodes * torch.mean(
            torch.sum(1 + 2 * log_var - mu.pow(2) - log_var.exp().pow(2), 1)
        )

        loss = kl_loss - dl_loss

        if self.add_diag:
            diag = []
            for i in range(Ad.shape[0]):
                diag.append(torch.diag(Ad[i, :]).unsqueeze(0))

            A = A + gama * torch.cat(diag, 0)
            # A = A + A * (gama * torch.eye(A.size(-1), device=A.device).unsqueeze(0))

        # A = laplacian_matrix(A, self_loop=True)
        A = self.laplacian_matrix(A, self_loop=True)
        # A = laplacian_batch(A.unsqueeze(3), True).squeeze()

        z_hat = gama.mean() * \
                mu.reshape(B, self.nodes, self.hidden) * \
                (1. - log_var.reshape(B, self.nodes, self.hidden))

        return A, gx, loss, z_hat

    @classmethod
    def laplacian_matrix(cls, A, self_loop=False):
        '''
        Computes normalized Laplacian matrix: A (B, N, N)
        '''
        if self_loop:
            A = A + torch.eye(A.size(1), device=A.device).unsqueeze(0)
        # deg_inv_sqrt = (A + 1e-5).sum(dim=1).clamp(min=0.001).pow(-0.5)
        deg_inv_sqrt = (torch.sum(A, 1) + 1e-5).pow(-0.5)

        LA = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)

        return LA


class GCN_Layer(nn.Module):
    def __init__(self, in_features, out_features, bnorm=True,
                 activation=nn.ReLU(), dropout=None):
        super(GCN_Layer, self).__init__()
        self.bnorm = bnorm
        fc = [nn.Linear(in_features, out_features)]
        if bnorm:
            fc.append(BatchNormGCN(out_features))
        if activation is not None:
            fc.append(activation)
        if dropout is not None:
            fc.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        x, A = data
        y = self.fc(torch.bmm(A, x))

        return [y, A]


def weight_xavier_init(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # nn.init.xavier_normal_(module.weight)
                nn.init.orthogonal_(module.weight)
                # nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class BatchNormGCN(nn.BatchNorm1d):
    """Batch normalization over GCN features"""

    def __init__(self, num_features):
        super(BatchNormGCN, self).__init__(num_features)

    def forward(self, x):
        return super(BatchNormGCN, self).forward(x.permute(0, 2, 1)).permute(0, 2, 1)
