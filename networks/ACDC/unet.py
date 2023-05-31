# coding=utf-8
# Partial codes inferenced from TransUNet (https://github.com/Beckschen/TransUNet)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import argparse
from os.path import join as pjoin
import torch
import torch.nn as nn
import torch.nn.functional as F
from . resnet_skip import ResNetV2
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, Conv3d, LayerNorm
from . import vit_seg_configs as configs
from . SPM import SPM


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class network(nn.Module):
    def __init__(self, in_channel=3, out_channel=2, training=True, config=None):
        super(network, self).__init__()
        self.dim = 512
        self.hybrid_model = ResNetV2(block_units=(2, 3, 5), width_factor=1)
        self.decoder_channels = (256, 128, 64, 16)
        self.skip_channels = [512, 256, 64, 16]
        channels = [16, 32, 64, 128, 256, 512]
        
        self.encoder1 = nn.Sequential(
            Conv3dReLU(in_channel, channels[0], kernel_size=3, padding=1),
            Conv3dReLU(channels[0], channels[0], kernel_size=3, padding=1)
        )
        
        self.decoder1 = nn.Sequential(
            Conv3dReLU(self.dim + self.skip_channels[0], self.decoder_channels[0], kernel_size=3, padding=1),
            Conv3dReLU(self.decoder_channels[0], self.decoder_channels[0], kernel_size=3, padding=1)
        )
        self.decoder2 = nn.Sequential(
            Conv3dReLU(self.decoder_channels[0] + self.skip_channels[1], self.decoder_channels[1], kernel_size=3, padding=1),
            Conv3dReLU(self.decoder_channels[1], self.decoder_channels[1], kernel_size=3, padding=1)
        )
        self.decoder3 = nn.Sequential(
            Conv3dReLU(self.decoder_channels[1] + self.skip_channels[2], self.decoder_channels[2], kernel_size=3, padding=1),
            Conv3dReLU(self.decoder_channels[2], self.decoder_channels[2], kernel_size=3, padding=1)
        )
        self.decoder4 = nn.Sequential(
            Conv3dReLU(self.decoder_channels[2] + channels[0], channels[0], kernel_size=3, padding=1),
            Conv3dReLU(channels[0], channels[0], kernel_size=3, padding=1)
        )
        
        self.down = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dw_up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        
        self.spm1 = SPM(config, self.skip_channels[0], (2, 2, 2))
        self.spm2 = SPM(config, self.skip_channels[1], (4, 4, 4))
        self.spm3 = SPM(config, self.skip_channels[2], (4, 8, 8))
        # self.cluster4 = SPM(config, channels[0], 16)
        
        self.learnable_shape_prior = nn.Parameter(torch.randn(1, out_channel, config.n_patches))

        self.segmentation_head = nn.Conv3d(channels[0], out_channel, kernel_size=3, padding=1)


    def forward(self, x):
        B = x.size()[0]
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1,1)
        t1 = self.encoder1(x)
        
        x, features = self.hybrid_model(t1)
        learnable_shape_prior = self.learnable_shape_prior.repeat(B,1,1)


        class_feature1, refined_shape_prior = self.spm1(features[0], learnable_shape_prior)
        x = self.up(x)
        x = torch.cat((x, class_feature1), 1)
        x = self.decoder1(x)


        class_feature2, refined_shape_prior = self.spm2(features[1], refined_shape_prior)
        x = self.up(x)
        x = torch.cat((x, class_feature2), 1)
        x = self.decoder2(x)
      

        class_feature3, refined_shape_prior = self.spm3(features[2], refined_shape_prior)
        x = self.dw_up(x)
        x = torch.cat((x, class_feature3), 1)
        x = self.decoder3(x)


        x = self.dw_up(x)
        x = torch.cat((x, t1), 1)
        x = self.decoder4(x)
        
        x = self.segmentation_head(x)

        return x


        
class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer.num_heads
        self.attention_head_size = int(config.n_patches / self.num_attention_heads)        # 768 / 6 = 128
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.n_patches, config.n_patches)
        self.key = Linear(config.n_patches, config.n_patches)
        self.value = Linear(config.n_patches, config.n_patches)

        self.out = Linear(config.n_patches, config.n_patches)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_attention_heads, config.n_classes, config.n_classes))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        attention_scores = attention_scores + self.position_embeddings                        # RPE
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        # weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.n_patches, config.hidden_size)
        self.fc2 = Linear(config.hidden_size, config.n_patches)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x       
        
        

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()

        self.attention_norm = LayerNorm(config.n_patches, eps=1e-6)
        self.ffn_norm = LayerNorm(config.n_patches, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x                                              
        x = self.attention_norm(x)                         
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x




class DecoderResBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv3 = Conv3dbn(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):

        feature_in = self.conv3(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + feature_in
        x = self.relu(x)

        return x



class Conv3dbn(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        # relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dbn, self).__init__(conv, bn)

class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)

