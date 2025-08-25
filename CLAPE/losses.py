# -*- coding: utf-8 -*-
# @Time         : 2025/8/24 18:00
# @Author       : Jue Wang
# @Description  : loss function

import torch
import torch.nn.functional as F
import torch.nn as nn

# Triplet center loss
# see https://github.com/xlliu7/Shrec2018_TripletCenterLoss.pytorch/blob/master/misc/custom_loss.py  # [batch,dim]
class TripletCenterLoss(nn.Module):
    # note the device
    def __init__(self, margin=0, num_classes=2, num_dim=2):
        super(TripletCenterLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.centers = nn.Parameter(torch.randn(num_classes, num_dim))  # random initialize as parameters

    def forward(self, inputs, targets):
        # resize inputs, delete labels with -1
        inputs = inputs.reshape(inputs.size(0) * inputs.size(1), inputs.size(2))
        targets = targets.reshape(targets.size(0) * targets.size(1))
        ignore_idx = targets != -1
        inputs = inputs[ignore_idx]
        targets = targets[ignore_idx]


        batch_size = inputs.size(0)
        targets_expand = targets.view(batch_size, 1).expand(batch_size, inputs.size(1))  # [batch, dim]
        centers_batch = self.centers.gather(0, targets_expand)  # embedding of index

        # compute pairwise distances between input features and corresponding centers
        centers_batch_bz = torch.stack([centers_batch] * batch_size)  # [batch, batch, dim]
        inputs_bz = torch.stack([inputs] * batch_size).transpose(0, 1)  # as above
        dist = torch.sum((centers_batch_bz - inputs_bz) ** 2, 2).squeeze()
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # for each anchor, find the hardest positive and negative (the furthest positive and nearest negative)
        # hard mining
        mask = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], []
        for i in range(batch_size):  # for each sample, we compute distance
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # mask[i]: positive samples of sample i
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # mask[i]==0: negative samples of sample i

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        # y_i = 1, means dist_an > dist_ap + margin will causes loss be zero
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

# CrossEntropy
class CrossEntropy(nn.Module):
    def forward(self, feature, label):
        feature = feature.reshape(feature.size(0) * feature.size(1), feature.size(2))
        label = label.reshape(label.size(0) * label.size(1))
        ignore_idx = label != -1
        feature = feature[ignore_idx]
        label = label[ignore_idx]
        num_class = feature.size(1)
        label = F.one_hot(label, num_class).float()

        # focal loss, alpha is weights above
        bc_loss = F.binary_cross_entropy(input=feature, target=label, reduction='mean')
        return bc_loss      
