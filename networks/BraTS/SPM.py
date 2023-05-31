# coding=utf-8
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


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class self_update_block(nn.Module):
    def __init__(self, config):
        super(self_update_block, self).__init__()
        num_layers = 2
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.n_patches, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, refined_shape_prior):
        for layer_block in self.layer:
            refined_shape_prior = layer_block(refined_shape_prior)

        encoded = self.encoder_norm(refined_shape_prior)
        
        return encoded

class cross_update_block(nn.Module):
    def __init__(self, n_class):
        super(cross_update_block, self).__init__()
        self.n_class = n_class
        self.softmax = Softmax(dim=-1)

    def forward(self, refined_shape_prior, feature):
        class_feature = torch.matmul(feature.flatten(2), refined_shape_prior.flatten(2).transpose(-1, -2))
        # scale
        class_feature = class_feature / math.sqrt(self.n_class)
        class_feature = self.softmax(class_feature)

        class_feature = torch.einsum("ijk, iklhw->ijlhw", class_feature, refined_shape_prior)
        class_feature = feature + class_feature
        return class_feature


        
class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer.num_heads
        self.attention_head_size = int(config.n_patches / self.num_attention_heads)
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

        # print(mixed_query_layer.shape)
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





class SPM(nn.Module):
    def __init__(self, config, in_channel, scale):
        super(SPM, self).__init__()
        self.scale = scale
        self.SUB = self_update_block(config)
        self.CUB  = cross_update_block(config.n_classes)
        self.resblock1 = DecoderResBlock(in_channel, in_channel)
        self.resblock2 = DecoderResBlock(in_channel, in_channel)
        self.resblock3 = DecoderResBlock(in_channel, config.n_classes)
        self.h = config.h
        self.w = config.w
        self.l = config.l
        self.dim = in_channel
        self.softmax = Softmax(dim=-1)

        
    def forward(self, feature, refined_shape_prior):
        b, n_class, _ = refined_shape_prior.size()
        B = feature.size()[0]
        # self update
        refined_shape_prior = self.SUB(refined_shape_prior)
        previous_class_center = refined_shape_prior
        refined_shape_prior = F.interpolate(refined_shape_prior.contiguous().view(b, n_class, self.h, self.w, self.l), scale_factor=self.scale, mode="trilinear")
        feature = self.resblock1(feature)
        feature = self.resblock2(feature)
        
        # cross update
        class_feature = self.CUB(refined_shape_prior, feature)
        

        refined_shape_prior = F.interpolate(self.resblock3(class_feature), scale_factor=(1.0 / self.scale), mode="trilinear").flatten(2) + previous_class_center

        return class_feature, refined_shape_prior



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
        # x = self.se_block(x)

        return x

