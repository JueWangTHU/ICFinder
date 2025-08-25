# -*- coding: utf-8 -*-
# @Time         : 2025/8/24 18:00
# @Author       : Jue Wang
# @Description  : Backbone models: MLP and CNN, and RNN and Transformer, not used in the study

import torch
import torch.nn as nn

# MLP
class ContinueModel(nn.Module):
    def __init__(self):
        super(ContinueModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GELU(),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
        )

        self.layer4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
        )

        self.layer5 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Softmax(-1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.layer1(x)
        inter = x  # dim 1024
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(self.layer5(x)), inter

# CNN
class CNNOD(nn.Module):
    def __init__(self):
        super(CNNOD, self).__init__()
        self.conv1 = nn.Conv1d(1280, 1280, kernel_size=7, stride=1, padding=3)
        self.norm1 = nn.BatchNorm1d(1280)
        self.conv2 = nn.Conv1d(1280, 128, kernel_size=5, stride=1, padding=2)
        self.norm2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 2, kernel_size=5, stride=1, padding=2)
        self.head = nn.Softmax(-1)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # batch, dim, length
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        inter = x.permute(0, 2, 1)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act(x)

        x = self.conv4(x)
        x = x.permute(0, 2, 1)
        return self.head(x), inter

# RNN
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.layer1 = nn.GRU(1280, 1280, 1, batch_first=True)
        self.layer2 = nn.GRU(1280, 128, 1, batch_first=True)
        self.layer3 = nn.GRU(128, 64, 1, batch_first=True)

        self.project = nn.Linear(64, 2)

        self.head = nn.Softmax(-1)

    def forward(self, x):
        device = x.device
        bz = x.size(0)
        h0 = torch.zeros(1, bz, 1024).to(device)
        c0 = torch.zeros(1, bz, 1024).to(device)
        out, _ = self.layer1(x, (h0, c0))
        inter = out

        h1 = torch.zeros(1, bz, 128).to(device)
        c1 = torch.zeros(1, bz, 128).to(device)
        out, _ = self.layer2(out, (h1, c1))

        h2 = torch.zeros(1, bz, 64).to(device)
        c2 = torch.zeros(1, bz, 64).to(device)
        out, _ = self.layer3(out, (h2, c2))

        x = self.project(out)
        return self.head(x), inter

#Transformer
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.begin = nn.Linear(1280, 1024)

        self.attention = nn.MultiheadAttention(1024, 8, 0.3, batch_first=True)

        self.norm = nn.LayerNorm(1024)
        self.dropout = nn.Dropout(0.3)

        self.ffn = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 1024)
        )

        self.class_head = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(-1)
        )

    def forward(self, x):
        x = self.begin(x)
        x, attention = self.attention(x, x, x)  # [b, length, length]
        x = self.dropout(self.norm(x + x))
        inter = x
        x = self.dropout(self.norm(self.ffn(x) + x))  # [length, 1024]
        return self.class_head(x), inter
