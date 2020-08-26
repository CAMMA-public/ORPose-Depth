from collections import OrderedDict

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn import init

def conv3x3(in_planes, out_planes, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding)

def conv7x7(in_planes, out_planes, stride=1, padding=3):
    """7x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=padding)


class DepthPose(nn.Module):
    def __init__(self):
        super(DepthPose, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2, 0)
        self.conv_x2_1 = conv3x3(64, 32)
        self.conv_x4_1 = conv3x3(64, 32)
        self.conv_x4_2 = conv3x3(32, 32)

        self.stage1_conv1 = conv3x3(1, 64)
        self.stage1_conv2 = conv3x3(64, 128)
        self.stage1_conv3 = conv3x3(128, 256)
        self.stage1_pixel_shuffle = nn.PixelShuffle(2)

        self.stage2_conv1 = conv3x3(64, 64)
        self.stage2_conv2 = conv3x3(64, 128)
        self.stage2_conv3 = conv3x3(128, 256)
        self.stage2_pixel_shuffle = nn.PixelShuffle(2)

        self.stage3_conv1 = conv3x3(64, 64)
        self.stage3_conv2 = conv3x3(64, 128)
        self.stage3_conv3 = conv3x3(128, 256)
        self.stage3_conv4 = conv3x3(256, 256)
        self.stage3_pixel_shuffle = nn.PixelShuffle(2)

        # for the pose estimation
        blocks = {}
        block0 = [{'conv1_1': [64, 64, 3, 1, 1]},
            {'conv1_2': [64, 64, 3, 1, 1]},
            {'pool1_stage1': [2, 2, 0]},
            {'conv2_1': [64, 128, 3, 1, 1]},
            {'conv2_2': [128, 128, 3, 1, 1]},
            {'pool2_stage1': [2, 2, 0]},
            {'conv3_1': [128, 256, 3, 1, 1]},
            {'conv3_2': [256, 256, 3, 1, 1]},
            {'conv3_3': [256, 256, 3, 1, 1]},
            {'conv3_4': [256, 256, 3, 1, 1]},
            {'pool3_stage1': [2, 2, 0]},
            {'conv4_1': [256, 512, 3, 1, 1]},
            {'conv4_2': [512, 512, 3, 1, 1]},
            {'conv4_3_CPM': [512, 256, 3, 1, 1]},
            {'conv4_4_CPM': [256, 128, 3, 1, 1]}]
        blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L1': [512, 38, 1, 1, 0]}]

        blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L2': [512, 19, 1, 1, 0]}]

        for i in range(2, 5):
            blocks['block%d_1' % i] = [
                {'Mconv1_stage%d_L1' % i: [249, 128, 7, 1, 3]},
                {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
                {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]}
            ]

            blocks['block%d_2' % i] = [
                {'Mconv1_stage%d_L2' % i: [249, 128, 7, 1, 3]},
                {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
                {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]}
            ]    

        models = {}
        models['block0'] = self._make_vgg19_block(block0)
        for k, v in blocks.items():
            models[k] = self._make_stages(list(v))

        self.model0 = models['block0']
        self.model1_1 = models['block1_1']
        self.model2_1 = models['block2_1']
        self.model3_1 = models['block3_1']
        self.model4_1 = models['block4_1']
        self.model1_2 = models['block1_2']
        self.model2_2 = models['block2_2']
        self.model3_2 = models['block3_2']
        self.model4_2 = models['block4_2']

        self._initialize_weights()

    def forward(self, x):
        saved_for_loss = []
        x = self.relu(self.stage1_conv1(x))
        x = self.relu(self.stage1_conv2(x))
        x_2 = self.stage1_pixel_shuffle(self.stage1_conv3(x)) # x_2 => 64x128x96

        x_2_down = self.relu(self.conv_x2_1(self.maxpool(x_2))) # x_2_down => 32x64x48

        # second stage
        x = self.relu(self.stage2_conv1(x_2))
        x = self.relu(self.stage2_conv2(x))
        x_4 = self.stage2_pixel_shuffle(self.stage2_conv3(x)) # x_4 => 64x256x192

        x4_down = self.relu(self.conv_x4_2(self.maxpool(self.relu(self.conv_x4_1(self.maxpool(x_4)))))) # x4_down => 32x64x48

        # third stage
        x = self.relu(self.stage3_conv1(x_4))
        x = self.relu(self.stage3_conv2(x))
        x = self.relu(self.stage3_conv3(x))
        x_8 = self.stage3_pixel_shuffle(self.stage3_conv4(x))


        x8_down = self.model0(x_8)                             # out1 => 128x64x96
        out1_1 = self.model1_1(x8_down)
        out1_2 = self.model1_2(x8_down)
        out2 = torch.cat((out1_1, out1_2, x8_down, x4_down, x_2_down), 1)
        saved_for_loss.append(out1_1)
        saved_for_loss.append(out1_2)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat((out2_1, out2_2, x8_down, x4_down, x_2_down), 1)
        saved_for_loss.append(out2_1)
        saved_for_loss.append(out2_2)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat((out3_1, out3_2, x8_down, x4_down, x_2_down), 1)
        saved_for_loss.append(out3_1)
        saved_for_loss.append(out3_2)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        saved_for_loss.append(out4_1)
        saved_for_loss.append(out4_2)

        return (out4_1, out4_2), (saved_for_loss)

    def _initialize_weights(self):
        init.orthogonal_(self.conv_x2_1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv_x4_1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv_x4_2.weight, init.calculate_gain('relu'))

        init.orthogonal_(self.stage1_conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.stage1_conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.stage1_conv3.weight)

        init.orthogonal_(self.stage2_conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.stage2_conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.stage2_conv3.weight)

        init.orthogonal_(self.stage3_conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.stage3_conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.stage3_conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.stage3_conv4.weight)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None: 
                    init.constant_(m.bias, 0.0)

        init.normal_(self.model1_1[8].weight, std=0.01)
        init.normal_(self.model1_2[8].weight, std=0.01)

        init.normal_(self.model2_1[12].weight, std=0.01)
        init.normal_(self.model3_1[12].weight, std=0.01)
        init.normal_(self.model4_1[12].weight, std=0.01)

        init.normal_(self.model2_2[12].weight, std=0.01)
        init.normal_(self.model3_2[12].weight, std=0.01)
        init.normal_(self.model4_2[12].weight, std=0.01)


    def _make_stages(self, cfg_dict):
        """Builds CPM stages from a dictionary
        Args:
            cfg_dict: a dictionary
        """
        layers = []
        for i in range(len(cfg_dict) - 1):
            one_ = cfg_dict[i]
            for k, v in one_.items():
                if 'pool' in k:
                    layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                            padding=v[2])]
                else:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                    kernel_size=v[2], stride=v[3],
                                    padding=v[4])
                    layers += [conv2d, nn.ReLU(inplace=True)]
        one_ = list(cfg_dict[-1].keys())
        k = one_[0]
        v = cfg_dict[-1][k]
        conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                        kernel_size=v[2], stride=v[3], padding=v[4])
        layers += [conv2d]
        return nn.Sequential(*layers)

    def _make_vgg19_block(self, block):
        """Builds a vgg19 block from a dictionary
        Args:
            block: a dictionary
        """
        layers = []
        for i in range(len(block)):
            one_ = block[i]
            for k, v in one_.items():
                if 'pool' in k:
                    layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                            padding=v[2])]
                else:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                    kernel_size=v[2], stride=v[3],
                                    padding=v[4])
                    layers += [conv2d, nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)    

def get_model():
    model = DepthPose()
    return model