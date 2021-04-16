# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email: houwen.peng@microsoft.com
# Main Results: see readme.md
# some code is based on updatenet: https://github.com/zhanglichao/updatenet/
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F

class SiamFC_(nn.Module):
    def __init__(self, feat_in=512, feature_out=512, anchor=5):
        super(SiamFC_, self).__init__()
        self.z_f = None  # for online tracking
        self.anchor = anchor
        self.feature_out = feature_out
        self.featureExtract = nn.Sequential(
            nn.Conv2d(3, 192, 11, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 512, 5),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(512, 768, 3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, 3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 512, 3),
            nn.BatchNorm2d(512),
        )
        self.conv_r1 = nn.Conv2d(feat_in, feature_out*4*anchor, 3)
        self.conv_r2 = nn.Conv2d(feat_in, feature_out, 3)
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out*2*anchor, 3)
        self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 3)
        self.regress_adjust = nn.Conv2d(4*anchor, 4*anchor, 1)

        self.r1_kernel = []
        self.cls1_kernel = []

        

    def forward(self, x):
        x_f = self.featureExtract(x)
        return self.regress_adjust(F.conv2d(self.conv_r2(x_f), self.r1_kernel)), \
               F.conv2d(self.conv_cls2(x_f), self.cls1_kernel)

    def feature_extractor(self, x):
        x_f = self.featureExtract(x)
        return x_f

    def kernel(self, z_f):
        self.z_f = z_f
        r1_kernel_raw = self.conv_r1(self.z_f)
        cls1_kernel_raw = self.conv_cls1(self.z_f)
        kernel_size = r1_kernel_raw.data.size()[-1]
        self.r1_kernel = r1_kernel_raw.view(self.anchor*4, self.feature_out, kernel_size, kernel_size)
        self.cls1_kernel = cls1_kernel_raw.view(self.anchor*2, self.feature_out, kernel_size, kernel_size)

    def template(self, z):
        self.z_f = self.feature_extractor(z)
        r1_kernel_raw = self.conv_r1(self.z_f)
        cls1_kernel_raw = self.conv_cls1(self.z_f)
        kernel_size = r1_kernel_raw.data.size()[-1]
        self.r1_kernel = r1_kernel_raw.view(self.anchor*4, self.feature_out, kernel_size, kernel_size)
        self.cls1_kernel = cls1_kernel_raw.view(self.anchor*2, self.feature_out, kernel_size, kernel_size)

    def track(self, x):
        xf = self.feature_extractor(x)
        score = self.connector(self.z_f, xf)
        return score

    def connector(self, template_feature, search_feature):
        pred_score = self.connect_model(template_feature, search_feature)
        return pred_score

