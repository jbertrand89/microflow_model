import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_
from src.models.model_utils.searaft_model_layers import BasicBlock, conv1x1, conv3x3
from src.models.model_utils.model_layers import conv, predict_flow, deconv, crop_like



def get_ResNetUpsampling_model(args):
    model = ResNetUpsampling(args, args.resnet_input_dim, args.resnet_imagenet_pretrained)
    return model


class ResNetUpsampling(nn.Module):
    """
    ResNet18, output resolution is 1/8.
    Each block has 2 layers.
    """
    def __init__(self, args, input_dim=3, init_weight=False):
        super().__init__()
        # Fixed hyper parameters
        norm_layer = nn.BatchNorm2d
        ratio = 1.0
        # Config
        self.args = args
        self.output_dim = args.searaft_dim * 2
        block = BasicBlock
        # block_dims = args.searaft_block_dims
        block_dims = [64, 128, 256]
        # print(type(block_dims))
        initial_dim = args.searaft_initial_dim
        self.init_weight = init_weight
        self.input_dim = input_dim
        # Class Variable
        self.in_planes = initial_dim
        for i in range(len(block_dims)):
            block_dims[i] = int(block_dims[i] * ratio)
        # Networks
        self.conv1 = nn.Conv2d(input_dim, initial_dim, kernel_size=7, stride=2, padding=3)
        self.bn1 = norm_layer(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        if args.searaft_pretrain == 'resnet34':
            n_block = [3, 4, 6]
        elif args.searaft_pretrain == 'resnet18':
            n_block = [2, 2, 2]
        else:
            raise NotImplementedError
        self.layer1 = self._make_layer(block, block_dims[0], stride=1, norm_layer=norm_layer, num=n_block[0])  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2, norm_layer=norm_layer, num=n_block[1])  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2, norm_layer=norm_layer, num=n_block[2])  # 1/8
        self.final_conv = conv1x1(block_dims[2], self.output_dim)

        self.init_conv = conv3x3(self.output_dim, self.output_dim)
        self.flow_head8 = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(args.searaft_dim, self.output_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_dim, 6, 3, padding=1)
        )

        self.flow_head4 = predict_flow(262, out_planes=6)
        self.flow_head2 = predict_flow(134, out_planes=6)
        self.flow_head1 = predict_flow(70, out_planes=6)

        self.upsampled_flow_8_to_4 = nn.ConvTranspose2d(6, 6, 4, 2, 1, bias=False)
        self.upsampled_flow_4_to_2 = nn.ConvTranspose2d(6, 6, 4, 2, 1, bias=False)
        self.upsampled_flow_2_to_1 = nn.ConvTranspose2d(6, 6, 4, 2, 1, bias=False)

        self.deconv_8_to_4 = deconv(256, 128)
        self.deconv_4_to_2 = deconv(128, 64)
        self.deconv_2_to_1 = deconv(64, 64)

        self._init_weights(args)

    def _init_weights(self, args):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.init_weight:
            from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
            if args.searaft_pretrain == 'resnet18':
                pretrained_dict = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
            else:
                pretrained_dict = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if self.input_dim == 6:
                for k, v in pretrained_dict.items():
                    if k == 'conv1.weight':
                        pretrained_dict[k] = torch.cat((v, v), dim=1)
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)

    def _make_layer(self, block, dim, stride=1, norm_layer=nn.BatchNorm2d, num=2):
        layers = []
        layers.append(block(self.in_planes, dim, stride=stride, norm_layer=norm_layer))
        for i in range(num - 1):
            layers.append(block(dim, dim, stride=1, norm_layer=norm_layer))
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # x 32, 6, 256, 256
        # ResNet Backbone
        x = self.relu(self.bn1(self.conv1(x)))  # after conv1 32, 64, 128, 128
        for i in range(len(self.layer1) - 1):
            x = self.layer1[i](x)
        out_conv_2 = self.layer1[len(self.layer1) - 1](x)  # after layer 1, resolution 1/2 :  32, 64, 128, 128
        x = self.layer2[0](out_conv_2)
        for i in range(1, len(self.layer2) - 1):
            x = self.layer2[i](x)
        out_conv_4 = self.layer2[len(self.layer2) - 1](x)  # after layer 2, resolution 1/4 :  32, 128, 64, 64
        x = self.layer3[0](out_conv_4)
        for i in range(1, len(self.layer3) - 1):
            x = self.layer3[i](x)
        out_conv_8 = self.layer3[len(self.layer3) - 1](x)  # after layer 3, resolution 1/8 :  32, 256, 32, 32

        # Output
        cnet = self.final_conv(out_conv_8)

        cnet_conv = self.init_conv(cnet)  # [32, 256, 32, 32]
        net, context = torch.split(cnet_conv, [self.args.searaft_dim, self.args.searaft_dim], dim=1)
        flow_and_info8 = self.flow_head8(net)  # [32, 6, 32, 32]
        flow8 = flow_and_info8[:, :2]  # [32, 2, 32, 32]
        # info8 = flow_and_info8[:, 2:]  # [32, 4, 32, 32]

        flow_and_info8_up    = crop_like(self.upsampled_flow_8_to_4(flow_and_info8), out_conv_4)  # 32, 6, 64, 64
        out_deconv_4 = crop_like(self.deconv_8_to_4(out_conv_8), out_conv_4)  # 32, 128, 64, 64
        concat4 = torch.cat((out_conv_4, out_deconv_4, flow_and_info8_up), 1)  # 32, 262, 64, 64
        flow_and_info4       = self.flow_head4(concat4)  # 32, 6, 64, 64
        flow4 = flow_and_info4[:, :2]

        flow_and_info4_up    = crop_like(self.upsampled_flow_4_to_2(flow_and_info4), out_conv_2)  # 32, 6, 128, 128
        out_deconv2 = crop_like(self.deconv_4_to_2(out_conv_4), out_conv_2)  # 32, 64, 128, 128
        concat2 = torch.cat((out_conv_2, out_deconv2, flow_and_info4_up), 1)  # 32, 134, 128, 128
        flow_and_info2 = self.flow_head2(concat2)  #32, 6, 128, 128]
        flow2 = flow_and_info2[:, :2]

        flow_and_info2_up = self.upsampled_flow_2_to_1(flow_and_info2)  # 32, 6, 256, 256
        out_deconv1 = self.deconv_2_to_1(out_conv_2)  # 32, 64, 256, 256
        concat1 = torch.cat((out_deconv1, flow_and_info2_up), 1)  # 32, 70, 256, 256
        flow_and_info1       = self.flow_head1(concat1)

        flow1 = flow_and_info1[:, :2]
        info = flow_and_info1[:, 2:]

        return flow1, flow2, flow4, flow8
