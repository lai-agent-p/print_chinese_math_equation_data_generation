# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
import pdb


def _bn_function_factory(norm, relu, conv):
    '''
    a function that generates function that processes a feature map
    :param norm: a normalization function
    :param relu: relu function
    :param conv: a convolution function
    :return: a function that apply layers listed above on tensors and
            concatenates them
    '''

    def bn_function(*inputs):
        '''
        apply batchnorm, relu activation and one more convolution
        layer to a feature map
        :param inputs: tensors as arguments
        :return: a tensor that contains processed and concatenated tensors
        '''
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, input_hidden, growth_rate, bn_rate, drop_rate, efficient=False):
        '''
        one layer in dense net
        :param input_hidden: hidden dimension of input feature
        :param growth_rate: how many filters to add each layer (`k` in paper)
        :param bn_rate: multiplicative factor for number of bottle neck layers
                       (i.e. bn_rate * growth_rate features in the bottleneck layer)
        :param drop_rate (float) - dropout rate after each dense layer
        :param efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower:
                           Pleiss, Geoff, et al. "Memory-efficient implementation of densenets."
                           arXiv preprint arXiv:1707.06990 (2017).
        '''
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(input_hidden)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(input_hidden, bn_rate * growth_rate,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_rate * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_rate * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        '''
        forward function for one layer on dense net

        :param prev_features: previous feature tensors passed in as arguments
        :return: new features
        '''
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        '''
            Explanation on efficient: notice that all three layers in bn_function: norm, relu, conv
            will produce a new feature map, and on training stage, all three feature maps will be saved.
            And those feature maps has big hidden size (prev_block_out_hidden + growth_rate * cur_layer_num_in_block).
            Saving all the feature maps is very memory inefficient. Thus, with cp.checkpoint, those
            feature maps will not be saved. Instead, they will be calculated again on backpropogation with previous
            layers. Of course it slows down the training process but it greatly reduce memory usage.
            Here is a comparison:
            each is a DenseNet-BC with 100 layers, batch size 64, tested on a NVIDIA Pascal Titan-X
            Naive	    memory consumption: 2.863	speed(s/minibatch): 0.165
            Efficient	memory consumption: 1.605	speed(s/minibatch): 0.207
            features from previous layer
            you can also check about the document for checkpoint:
            https://pytorch.org/docs/stable/checkpoint.html
        '''
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        bottleneck_output = F.dropout(bottleneck_output, p=self.drop_rate, training=self.training)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, input_hidden, output_hidden, drop_rate):
        '''
        bottle neck layer between dense blocks,
        Notice that this class inherits nn.sequential, thus it doesn't
        have forward function
        :param input_hidden: hidden size of input feature
        :param output_hidden: hidden size of output feature
        '''
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(input_hidden))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(input_hidden, output_hidden,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('dropout', nn.Dropout(drop_rate))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _Transition_no_pool(nn.Sequential):
    def __init__(self, input_hidden, output_hidden):
        '''
        bottle neck layer between dense blocks,
        Notice that this class inherits nn.sequential, thus it doesn't
        have forward function
        :param input_hidden: hidden size of input feature
        :param output_hidden: hidden size of output feature
        '''
        super(_Transition_no_pool, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(input_hidden))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(input_hidden, output_hidden,
                                          kernel_size=1, stride=1, bias=False))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, input_hidden, bn_rate, growth_rate, drop_rate, efficient=False):
        '''
        :param num_layers: number of dense layers in a block
        :param input_hidden: hidden dimension of input data
        :param bn_rate: multiplicative factor for number of bottle neck layers
                       (i.e. bn_rate * growth_rate features in the bottleneck layer)
        :param growth_rate: how many filters to add each layer (`k` in paper)
        :param drop_rate:(float) dropout rate after each dense layer
        :param efficient:  (bool) - set to True to use checkpointing. Much more memory efficient, but slower:
                           Pleiss, Geoff, et al. "Memory-efficient implementation of densenets."
                           arXiv preprint arXiv:1707.06990 (2017).
        '''
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                input_hidden + i * growth_rate,
                growth_rate=growth_rate,
                bn_rate=bn_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        '''
        forward function for a dense block, all output feature of
        dense_layers in the block are saved in a list. It's only
        concatenated to one tensor when the program finished run
        dense layer.
        :param init_features: feature from input layer or
        previous dense blocks
        :return: one feature map of size (batch_size, prev_hidden + growth_rate*num_layers, h', w')
        '''
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    For each dense layer, the model first compress the input feature to a feature map with hidden size
    growth_rate * bn_rate, then use this map to get the feature map with hidden size growth_rate. The
    feature map with hidden size growth_rate will be appended to the output feature.
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each dense block
        init_hidden (int) - the hidden dimension for convolutional layer
        bn_rate (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_rate * growth_rate features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        efficient (bool) - set to True to use checkpointing( a new utility function in pytorch).
                           Much more memory efficient, but slower:
                           Pleiss, Geoff, et al. "Memory-efficient implementation of densenets."
                           arXiv preprint arXiv:1707.06990 (2017).
    """

    def __init__(self, growth_rate=3, block_config=(8, 8, 8), compression=0.5,
                 init_hidden=24, bn_rate=4, drop_rate=0, encoder_out_dim = 128, efficient=False, use_dense_mdlstm=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 7

        # First convolution, compress before feeding features to dense networks

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, init_hidden, kernel_size=3, stride=1, padding=0, bias=False)),
        ]))
        self.features.add_module('norm0', nn.BatchNorm2d(init_hidden))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('pool0', nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                                                       ceil_mode=False))

        # Each denseblock
        num_features = init_hidden
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                input_hidden=num_features,
                bn_rate=bn_rate,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                # compress
                trans = _Transition(input_hidden=num_features,
                                    output_hidden=int(num_features * compression),
                                    drop_rate=drop_rate)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)
            elif use_dense_mdlstm:
                trans = _Transition_no_pool(input_hidden=num_features,
                                            output_hidden=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        self.out_num_features = num_features
        # Final batch norm
        # self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                # hand implementation of xavier initialization
                # https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                # initialization for gamma in paper
                # assuming batch norm is the normalzation tool
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                # initialization for beta in paper
                # assuming batch norm is the normalzation tool
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

        self.out_projection = nn.Linear(num_features, encoder_out_dim)

    def forward(self, x):
        '''
        run densenet model, based on original implementation on tensorflow,
        the final feature map is not normalzed nor activated
        :param x: input image as float tensor with shape[batch_size, 1, h, w]
        :return: output feature after go through densenet (batch_size, feature_size, h', w')
        '''
        features = self.features(x)
        out_feature = self.out_projection(torch.mean(features, dim=(2, 3)))
        return out_feature
