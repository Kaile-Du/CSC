import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, start, end):
        super(GraphConvolution, self).__init__()
        num_nodes = end
        self.general_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.general_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)
        self.conv_create_co_mat = nn.Conv1d(in_features*2, num_nodes, 1)
        self.specific_weight = nn.Conv1d(in_features, out_features, 1) # W_d
        self.specific_adj_show = None
        self.grads = {}

    def forward_general_gcn(self, x):
        x = self.general_adj(x.transpose(1, 2))
        x = self.general_weight(x.transpose(1, 2))
        return x

    def forward_construct_specific_graph(self, x):
        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))
        x = torch.cat((x_glb, x), dim=1)
        specific_adj = self.conv_create_co_mat(x) 
        specific_adj = torch.sigmoid(specific_adj)
        return specific_adj

    def forward_specific_gcn(self, x, specific_adj):
        x = torch.matmul(x, specific_adj)
        x = self.relu(x)
        x = self.specific_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        out_general = self.forward_general_gcn(x)
        x = x + out_general
        specific_adj = self.forward_construct_specific_graph(x)
        self.specific_adj_show = specific_adj
        x = self.forward_specific_gcn(x, specific_adj)
        return x


class CI_GCN(nn.Module):
    def __init__(self, model, start, end):
        super(CI_GCN, self).__init__()
        self.model = model
        self.features = nn.Sequential(
            model.space_to_depth,
            model.conv1,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.task_size = end - start
        self.fc = nn.Conv2d(model.head.fc.in_features, end - start, (1,1), bias=False)
        self.conv_transform = nn.Conv2d(2048, 1024, (1,1))
        self.relu = nn.LeakyReLU(0.2)
        self.gcn = GraphConvolution(1024, 1024, start, end)
        self.mask_mat = nn.Parameter(torch.eye(self.task_size).float())
        self.last_linear = nn.Conv1d(1024, self.task_size, 1)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward_feature(self, x):
        x = self.features(x)
        return x

    def forward_classification(self, x):
        x = self.fc(x) 
        x = x.view(x.size(0), x.size(1), -1)
        x = x.topk(1, dim=-1)[0].mean(dim=-1)
        return x

    def mask(self, x):
        mask = self.fc(x) 
        mask = mask.view(mask.size(0), mask.size(1), -1)
        mask  = F.softmax(mask, dim=2)
        mask = mask.transpose(1, 2)
        x = self.conv_transform(x) 
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)
        return x 

    def forward_gcn(self, x):
        x = self.gcn(x)
        return x

    def forward(self, x):
        x = self.forward_feature(x)
        out1 = self.forward_classification(x)
        v_0 = self.mask(x) 
        v_2 = self.forward_gcn(v_0)
        v_2 = v_0 + v_2
        out2 = self.last_linear(v_2) 
        mask_mat = self.mask_mat.detach()
        out2 = (out2 * mask_mat).sum(-1)

        return out1, out2, (out1+out2)/2

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.parameters())
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': large_lr_layers, 'lr': lr},
                ]

