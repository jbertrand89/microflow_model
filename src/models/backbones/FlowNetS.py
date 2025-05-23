import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from src.models.model_utils.model_layers import conv, predict_flow, deconv, crop_like
import torch.nn.functional as F


def get_FlowNetS_model(data=None, batchNorm=False):
    model = FlowNetS(batchNorm=batchNorm)
    return model


class FlowNetS(nn.Module):
    """
    FlowNetS implementation from Clement Pinard
    https://github.com/ClementPinard/FlowNetPytorch/blob/master/models/FlowNetS.py

    The model is predicted at level h/4 of the original image, and interpolated to the original image size (with no learnable upsampling)
    """
    expansion = 1

    def __init__(self, batchNorm=True):
        super(FlowNetS, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, 2, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        # x = 1*2*256*256  (batch*channel*h*w)
        # print(f"x {x.shape}")
        out_conv2 = self.conv2(self.conv1(x))  # 1*128*64*64
        # print(f"out_conv2 {out_conv2.shape}")
        out_conv3 = self.conv3_1(self.conv3(out_conv2))  # 1*256*32*32
        out_conv4 = self.conv4_1(self.conv4(out_conv3))  # 1*512*16*16
        out_conv5 = self.conv5_1(self.conv5(out_conv4))  # 1*512*8*8
        out_conv6 = self.conv6_1(self.conv6(out_conv5))  # 1*1024*4*4

        flow6 = self.predict_flow6(out_conv6)  # 1*2*
        # upsampled to 1*2*128*128 and no crop
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)  # 1*2*
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)  #1*512*

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)  #1*1026*
        flow5 = self.predict_flow5(concat5)  #1*2*
        flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)  #1*2*
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)  # 1*256*

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)  # 1*770*
        flow4 = self.predict_flow4(concat4)  # 1*2*
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)  # 1*2*
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)  # 1 * 128 *

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)  # 1*386*
        flow3 = self.predict_flow3(concat3)  # 1*2*
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)  # 1*2*
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)  #1*194
        flow2 = self.predict_flow2(concat2)  #1*2*64*64

        # Upsample to dimension of the original
        _, _, h, w = x.shape
        flow1 = F.interpolate(flow2, size=(h, w), mode='bilinear', align_corners=False)
        return flow1, flow3, flow4, flow5, flow6

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]



