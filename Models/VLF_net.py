import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,dilation1 = 1,dilation2 = 1):
        super(ResnetBlock, self).__init__()
        self.model = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias,dilation1 = dilation1,dilation2 = dilation2)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias,dilation1 = 1,dilation2 = 1):

        self.Block = nn.Sequential(OrderedDict([]))

        p = 0
        if padding_type == 'reflect':
            #镜像填充  ReflectionPad2d
            self.Block.add_module('padding1', nn.ReflectionPad2d(dilation1))
        elif padding_type == 'replicate':
            #重复填充  ReplicationPad2d
            self.Block.add_module('padding1', nn.ReplicationPad2d(dilation1))
        elif padding_type == 'zero':
            p = dilation1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.Block.add_module('conv1', nn.Conv2d(dim, dim, kernel_size=3, padding=p,dilation =dilation1, bias=use_bias))
        self.Block.add_module('norm1', norm_layer(dim))
        self.Block.add_module('relu1', nn.ReLU(inplace=True))
        if use_dropout:
            self.Block.add_module('dropout', nn.Dropout(0.5))

        p = 0
        if padding_type == 'reflect':
            self.Block.add_module('padding2', nn.ReflectionPad2d(dilation2))
          
        elif padding_type == 'replicate':
            self.Block.add_module('padding2', nn.ReplicationPad2d(dilation2))
        elif padding_type == 'zero':
            p = dilation2
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.Block.add_module('conv2', nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=dilation2,bias=use_bias))
        self.Block.add_module('norm2', norm_layer(dim))
        
        return self.Block

    def forward(self, x):
        x1 = self.model(x)
        out = x + x1
        return out


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function

class VLF_Layer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):

        super(VLF_Layer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, int(bn_size *
                        growth_rate), kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(int(bn_size * growth_rate))),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(int(bn_size * growth_rate), growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        new_features = new_features + prev_features[-1]

        return new_features


class VLF_Block(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        '''
        :param num_layers:  the number of layers in this block 多少层网络
        :param num_input_features: the number of the input channel 初始输入通道
        :param bn_size:  bottonneck， bn_size * k features in the bottleneck layer  降维作用
        :param growth_rate:  how many filters to add each layer  网络增长率
        :param drop_rate:  dropout rate after each VLF-block
        :param  efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
        '''
        super(VLF_Block, self).__init__()
        for i in range(num_layers):
            layer = VLF_Layer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('VLF-layer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)



class Transition_layer(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition_layer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2,ceil_mode=True))



class VLF_net(nn.Module):
    """
    Vehicle Logo Feature Extraction Network
    Args:
        growth_rate (int) - how many filters to add each layer
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each VLF-block
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
        compression: compression factor for each transition layer  中间的降维度的值
    """
    def __init__(self, growth_rate=16, block_config=(3,3,3,3,3), compression=0.5,
                 num_init_features=32, bn_size=0.5, drop_rate=0,
                 num_classes=44, small_inputs=False, efficient=False):

        super(VLF_net, self).__init__()
        assert 0 < compression <= 1, 'compression of VLF-net should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=False))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=1, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=False))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i == 0 :
                bn_size = 1
            else:
                bn_size = 0.5

            growth_rate = growth_rate*2
            # Each VLF-block
            block = VLF_Block(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('VLF-block%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition_layer(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)
                reblock = ResnetBlock(dim = num_features, padding_type = "zero", norm_layer = nn.BatchNorm2d, use_dropout = True, use_bias = False)

                self.features.add_module('resBlock%d' % (i + 1), reblock)
                # self.features.add_module('pool%d' % (i + 1), nn.AvgPool2d(kernel_size=2, stride=2,ceil_mode=True))

            else:
                trans = Transition_layer(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        self.features.add_module('conv-1', nn.Conv2d(num_features, int(num_features*0.5), kernel_size=3, stride=1, padding=1, bias=False))
        self.features.add_module('norm-1', nn.BatchNorm2d(int(num_features*0.5)))
        self.features.add_module('relu-1', nn.ReLU(inplace=False))

        self.features.add_module('conv-2', nn.Conv2d(int(num_features*0.5),num_features, kernel_size=3, stride=1, padding=1, bias=False))
        self.features.add_module('norm-2', nn.BatchNorm2d(num_features))
        self.features.add_module('relu-2', nn.ReLU(inplace=False))

        self.features.add_module('conv-3', nn.Conv2d(num_features, int(num_features*0.5), kernel_size=3, stride=1, padding=1, bias=False))
        self.features.add_module('norm-3', nn.BatchNorm2d(int(num_features*0.5)))
        self.features.add_module('relu-3', nn.ReLU(inplace=False))

        # Linear layer
        self.classifier = nn.Linear(int(num_features*0.5), num_classes)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=False)

        out = F.avg_pool2d(out, kernel_size=(out.shape[2],out.shape[3]),ceil_mode=True).view(features.size(0), -1)
        out = self.classifier(out)
        return out





class CCML_VLF_net(nn.Module):
    """
    VLF-net-based Category-Consistent Deep Network
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each VLF-block
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
        compression: compression factor for each transition layer  中间的降维度的值
    """
    def __init__(self, growth_rate=16, block_config=(3,3,3,3,3), compression=0.5,
                 num_init_features=32, bn_size=0.5, drop_rate=0,
                 num_classes=44, small_inputs=False, efficient=False,):

        super(CCML_VLF_net, self).__init__()
        assert 0 < compression <= 1, 'compression of VLF-net should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=False))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=False))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i == 0 :
                bn_size = 1
            else:
                bn_size = 0.5

            growth_rate = growth_rate*2
            # Each VLF-block
            block = VLF_Block(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('VLF-block%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition_layer(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)
                reblock = ResnetBlock(dim = num_features, padding_type = "zero", norm_layer = nn.BatchNorm2d, use_dropout = True, use_bias = False)

                self.features.add_module('resBlock%d' % (i + 1), reblock)

            else:
                trans = Transition_layer(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        self.features.add_module('conv-1', nn.Conv2d(num_features, int(num_features*0.5), kernel_size=3, stride=1, padding=1, bias=False))
        self.features.add_module('norm-1', nn.BatchNorm2d(int(num_features*0.5)))
        self.features.add_module('relu-1', nn.ReLU(inplace=False))

        self.features.add_module('conv-2', nn.Conv2d(int(num_features*0.5),num_features, kernel_size=3, stride=1, padding=1, bias=False))
        self.features.add_module('norm-2', nn.BatchNorm2d(num_features))
        self.features.add_module('relu-2', nn.ReLU(inplace=False))

        self.features.add_module('conv-3', nn.Conv2d(num_features, int(num_features*0.5), kernel_size=3, stride=1, padding=1, bias=False))
        self.features.add_module('norm-3', nn.BatchNorm2d(int(num_features*0.5)))
        self.features.add_module('relu-3', nn.ReLU(inplace=False))


        # Linear layer
        self.classifier = nn.Linear(int(num_features*0.5), num_classes)

        # mask layer
        self._mask_out = nn.Sequential()
        self._mask_out.add_module('conv-4', nn.Conv2d(int(num_features * 0.5), num_classes, kernel_size=3, stride=1, padding=1, bias=False))
        self._mask_out.add_module('up-4',         nn.UpsamplingBilinear2d(scale_factor=2))

        self._mask_out.add_module('norm-4', nn.BatchNorm2d(num_classes))
        self._mask_out.add_module('relu-4', nn.ReLU(True))
        self._mask_out.add_module('conv-5', nn.Conv2d(num_classes,num_classes, kernel_size=3, stride=1, padding=1, bias=False))
        self._mask_out.add_module('conv-6', nn.Conv2d(num_classes,1, kernel_size=1, stride=1, padding=0, bias=False))
        self._mask_out.add_module('sigmoid-6', nn.Sigmoid())


        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x,Test=False):
        features = self.features(x)
        out = F.relu(features, inplace=False)
        out_fc = F.avg_pool2d(out, kernel_size=(out.shape[2],out.shape[3]),ceil_mode=True).view(features.size(0), -1)
        out = self.classifier(out_fc)
        if Test ==True:
            return out

        outlist = []
        outlist.append(out)
        maskout = self._mask_out(features)
        outlist.append(maskout)
        return outlist
