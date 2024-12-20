from torch.utils.data import Dataset
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image



class VGG_FeatureExtractor(nn.Module):
    """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

    def __init__(self, input_channel, output_channel=512):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x16x50
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128x8x25
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),  # 256x8x25
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x25
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),  # 512x4x25
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x25
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))  # 512x1x24

    def forward(self, input):
        return self.ConvNet(input)


class RCNN_FeatureExtractor(nn.Module):
    """ FeatureExtractor of GRCNN (https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf) """

    def __init__(self, input_channel, output_channel=512):
        super(RCNN_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64 x 16 x 50
            GRCL(self.output_channel[0], self.output_channel[0], num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, 2),  # 64 x 8 x 25
            GRCL(self.output_channel[0], self.output_channel[1], num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, (2, 1), (0, 1)),  # 128 x 4 x 26
            GRCL(self.output_channel[1], self.output_channel[2], num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, (2, 1), (0, 1)),  # 256 x 2 x 27
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 2, 1, 0, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True))  # 512 x 1 x 26

    def forward(self, input):
        return self.ConvNet(input)


class ResNet_FeatureExtractor(nn.Module):
    """ FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """

    def __init__(self, input_channel, output_channel=512):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel)#, BasicBlock, [1, 2, 5, 3])

    def forward(self, input):
        return self.ConvNet(input)


# For Gated RCNN
class GRCL(nn.Module):

    def __init__(self, input_channel, output_channel, num_iteration, kernel_size, pad):
        super(GRCL, self).__init__()
        self.wgf_u = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False)
        self.wgr_x = nn.Conv2d(output_channel, output_channel, 1, 1, 0, bias=False)
        self.wf_u = nn.Conv2d(input_channel, output_channel, kernel_size, 1, pad, bias=False)
        self.wr_x = nn.Conv2d(output_channel, output_channel, kernel_size, 1, pad, bias=False)

        self.BN_x_init = nn.BatchNorm2d(output_channel)

        self.num_iteration = num_iteration
        self.GRCL = [GRCL_unit(output_channel) for _ in range(num_iteration)]
        self.GRCL = nn.Sequential(*self.GRCL)

    def forward(self, input):
        """ The input of GRCL is consistant over time t, which is denoted by u(0)
        thus wgf_u / wf_u is also consistant over time t.
        """
        wgf_u = self.wgf_u(input)
        wf_u = self.wf_u(input)
        x = F.relu(self.BN_x_init(wf_u))

        for i in range(self.num_iteration):
            x = self.GRCL[i](wgf_u, self.wgr_x(x), wf_u, self.wr_x(x))

        return x


class GRCL_unit(nn.Module):

    def __init__(self, output_channel):
        super(GRCL_unit, self).__init__()
        self.BN_gfu = nn.BatchNorm2d(output_channel)
        self.BN_grx = nn.BatchNorm2d(output_channel)
        self.BN_fu = nn.BatchNorm2d(output_channel)
        self.BN_rx = nn.BatchNorm2d(output_channel)
        self.BN_Gx = nn.BatchNorm2d(output_channel)

    def forward(self, wgf_u, wgr_x, wf_u, wr_x):
        G_first_term = self.BN_gfu(wgf_u)
        G_second_term = self.BN_grx(wgr_x)
        G = F.sigmoid(G_first_term + G_second_term)

        x_first_term = self.BN_fu(wf_u)
        x_second_term = self.BN_Gx(self.BN_rx(wr_x) * G)
        x = F.relu(x_first_term + x_second_term)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


# Todo: transform to the pretrained Resnet 50 or smaller
# class ResNet(nn.Module):

#     def __init__(self, input_channel, output_channel, block, layers):
#         super(ResNet, self).__init__()

#         self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

#         self.inplanes = int(output_channel / 8)
#         self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16),
#                                  kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
#         self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes,
#                                  kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn0_2 = nn.BatchNorm2d(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)

#         self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
#         self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[
#                                0], kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

#         self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
#         self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[
#                                1], kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

#         self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
#         self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
#         self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[
#                                2], kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

#         self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
#         self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
#                                  3], kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
#         self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
#         self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
#                                  3], kernel_size=2, stride=1, padding=0, bias=False)
#         self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv0_1(x)
#         x = self.bn0_1(x)
#         x = self.relu(x)
#         x = self.conv0_2(x)
#         x = self.bn0_2(x)
#         x = self.relu(x)

#         x = self.maxpool1(x)
#         x = self.layer1(x)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         x = self.maxpool2(x)
#         x = self.layer2(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)

#         x = self.maxpool3(x)
#         x = self.layer3(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu(x)

#         x = self.layer4(x)
#         x = self.conv4_1(x)
#         x = self.bn4_1(x)
#         x = self.relu(x)
#         x = self.conv4_2(x)
#         x = self.bn4_2(x)
#         x = self.relu(x)

#         return x

from torchvision import models

class ResNet(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ResNet, self).__init__()

        # Load pretrained ResNet-18 model
        self.resnet18 = models.resnet18(pretrained=True)

        # Modify the first convolution layer to accept input_channel (e.g., 1 for grayscale)
        if input_channel != 3:  # Only replace if input channels aren't RGB
            self.resnet18.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Freeze the pretrained layers (optional, for fine-tuning)
        for param in self.resnet18.parameters():
            param.requires_grad = True  # Set to False to freeze layers

        # Remove the fully connected (fc) layer for custom output
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])  # Remove avgpool & fc layers

        # Add final layers to adapt to the desired output channels
        self.conv_final = nn.Conv2d(512, output_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_final = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.resnet18(x)  # Pass through ResNet-18 backbone
        x = self.conv_final(x)  # Adjust channel dimension
        x = self.bn_final(x)
        x = self.relu(x)
        return x



class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38): # 62
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, is_train=True, batch_max_length=25):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=64)# self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
            probs = self.generator(output_hiddens)

        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
        concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, 64)# num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38): #62
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, is_train=True, batch_max_length=25):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=64)#self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
            probs = self.generator(output_hiddens)

        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
        concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha


class TPS_SpatialTransformerNetwork(nn.Module):
    """ Rectification Network of RARE, namely TPS based STN """

    def __init__(self, F, I_size, I_r_size, I_channel_num=1):
        """ Based on RARE TPS
        input:
            batch_I: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
            I_size : (height, width) of the input image I
            I_r_size : (height, width) of the rectified image I_r
            I_channel_num : the number of channels of the input image I
        output:
            batch_I_r: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
        """
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size  # = (I_r_height, I_r_width)
        self.I_channel_num = I_channel_num
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def forward(self, batch_I):
        batch_C_prime = self.LocalizationNetwork(batch_I)  # batch_size x K x 2
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)  # batch_size x n (= I_r_width x I_r_height) x 2
        build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])

        if torch.__version__ > "1.2.0":
            batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border', align_corners=True)
        else:
            batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border')

        return batch_I_r


class LocalizationNetwork(nn.Module):
    """ Localization Network of RARE, which predicts C' (K x 2) from I (I_width x I_height) """

    def __init__(self, F, I_channel_num):
        super(LocalizationNetwork, self).__init__()
        self.F = F
        self.I_channel_num = I_channel_num
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.I_channel_num, out_channels=64, kernel_size=3, stride=1, padding=1,
                      bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 64 x I_height/2 x I_width/2
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 128 x I_height/4 x I_width/4
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 256 x I_height/8 x I_width/8
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)  # batch_size x 512
        )

        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.F * 2)

        # Init fc2 in LocalizationNetwork
        self.localization_fc2.weight.data.fill_(0)
        """ see RARE paper Fig. 6 (a) """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, batch_I):
        """
        input:     batch_I : Batch Input Image [batch_size x I_channel_num x I_height x I_width]
        output:    batch_C_prime : Predicted coordinates of fiducial points for input batch [batch_size x F x 2]
        """
        batch_size = batch_I.size(0)
        features = self.conv(batch_I).view(batch_size, -1)
        batch_C_prime = self.localization_fc2(self.localization_fc1(features)).view(batch_size, self.F, 2)
        return batch_C_prime


class GridGenerator(nn.Module):
    """ Grid Generator of RARE, which produces P_prime by multipling T with P """

    def __init__(self, F, I_r_size):
        """ Generate P_hat and inv_delta_C for later """
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)  # F x 2
        self.P = self._build_P(self.I_r_width, self.I_r_height)
        ## for multi-gpu, you need register buffer
        self.register_buffer("inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C)).float())  # F+3 x F+3
        self.register_buffer("P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float())  # n x F+3
        ## for fine-tuning with different image width, you may use below instead of self.register_buffer
        #self.inv_delta_C = torch.tensor(self._build_inv_delta_C(self.F, self.C)).float().cuda()  # F+3 x F+3
        #self.P_hat = torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float().cuda()  # n x F+3

    def _build_C(self, F):
        """ Return coordinates of fiducial points in I_r; C """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2

    def _build_inv_delta_C(self, F, C):
        """ Return inv_delta_C which is needed to calculate T """
        hat_C = np.zeros((F, F), dtype=float)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)
        # print(C.shape, hat_C.shape)
        delta_C = np.concatenate(  # F+3 x F+3
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)  # 1 x F+3
            ],
            axis=0
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # F+3 x F+3

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width  # self.I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height  # self.I_r_height
        P = np.stack(  # self.I_r_width x self.I_r_height x 2
            np.meshgrid(I_r_grid_x, I_r_grid_y),
            axis=2
        )
        return P.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3

    def build_P_prime(self, batch_C_prime):
        """ Generate Grid from batch_C_prime [batch_size x F x 2] """
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(
            batch_size, 3, 2).float().to(device)), dim=1)  # batch_size x F+3 x 2
        batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)  # batch_size x F+3 x 2
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)  # batch_size x n x 2
        return batch_P_prime  # batch_size x n x 2


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Model(nn.Module):

    def __init__(self, Transformation, FeatureExtraction, SequenceModeling, Prediction,
                 num_fiducial, imgH, imgW, input_channel, output_channel, hidden_size, num_class, batch_max_length):
        """ Transformation: FCN or None
            FeatureExtraction: VGG or RCNN
            SequenceModeling: BiLSTM
            Prediction: CTC or Attention
            num_fiducial = 20 default
        """
        super(Model, self).__init__()

        self.batch_max_length = batch_max_length
        self.stages = {'Trans': Transformation, 'Feat': FeatureExtraction,
                       'Seq': SequenceModeling, 'Pred': Prediction}

        """ Transformation """
        if Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=num_fiducial, I_size=(imgH, imgW), I_r_size=(imgH, imgW), I_channel_num=input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        elif FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(input_channel, output_channel)
        elif FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
                BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
            self.SequenceModeling_output = hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)
        elif Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, hidden_size, num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.batch_max_length)

        return prediction



class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


class IAMWordsDataset(Dataset):
    def __init__(self, root_dir, words_txt_path, transform=None, characters='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
        """
        Args:
            root_dir (str): Directory containing word images.
            words_txt_path (str): Path to words.txt file.
            transform (callable, optional): Transform to apply to images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.characters = characters

        # Read the words.txt file and extract image paths and labels
        with open(words_txt_path, 'r') as f:
            for line in f:
                if line.startswith("#") or "err" in line:  # Skip comments or erroneous lines
                    continue
                parts = line.strip().split()
                file_id = parts[0]  # e.g., "a01-000u-00-00"
                file_id_x = file_id[:3]
                label = " ".join(parts[8:]).lower() # The word transcription
                #consider_only_text = ['from', 'the', 'a', 'turn', 'to', 'and', 'each', 'would', 'they', 'put', 'now', 'put', 'should', 'take']
                file_path = os.path.join(root_dir, file_id_x, "-".join(file_id.split("-")[:2]), f"{file_id}.png")
                if os.path.exists(file_path):# and label in consider_only_text:
                    self.data.append((file_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]

        # Load the image
        image = Image.open(file_path).convert("L")  # Convert to grayscale
        if self.transform:
            image = self.transform(image)

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.characters}]'
            label = re.sub(out_of_char, '', label)

        return image, label

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output