import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from src.models.model_utils.model_layers import conv, predict_flow, deconv, crop_like, correlate


def get_GeoFlowNetC_model(batchNorm=False):
    model = GeoFlowNetC(batchNorm=batchNorm)
    return model


class GeoFlowNetC(nn.Module):
    """
    GeoFlowNetC is a modified version of FlowNetC and GeoFlowNet.
    It uses a correlation layer to compute the correlation between the two images like in FlowNetC, 
    but the upsampling is done with learnable parameters, and the architecture is more similar to GeoFlowNet.
    """
    expansion = 1

    def __init__(self, batchNorm=True):
        super(GeoFlowNetC,self).__init__()

        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,   1,   64)
        self.conv2   = conv(self.batchNorm,  64,  128)
        self.conv3   = conv(self.batchNorm, 128,  256, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv_redir = conv(self.batchNorm, 512, 32, kernel_size=1, stride=1)

        self.conv4_1 = conv(self.batchNorm, 473,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

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
        x1 = x[:, :1]
        x2 = x[:, 1:]

        # x1:  b x 1 x 256 x 256
        out_conv1a = self.conv1(x1)  # b x 64 x 256 x 256
        out_conv2a = self.conv2(out_conv1a)  # b x 128 x 256 x 256
        out_conv3a = self.conv3_1(self.conv3(out_conv2a))  # b x 256 x 128 x 128
        out_conv4a = self.conv4(out_conv3a)  # b x 512 x 64 x 64

        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3_1(self.conv3(out_conv2b))
        out_conv4b = self.conv4(out_conv3b)  # b x 512 x 64 x 64

        out_conv_redir = self.conv_redir(out_conv4a)  # b x 32 x 64 x 64
        out_correlation = correlate(out_conv4a, out_conv4b)  # b x 441 x 64 x 64

        in_conv4_1 = torch.cat([out_conv_redir, out_correlation], dim=1)  # b x 473 x 64 x 64
        out_conv4 = self.conv4_1(in_conv4_1)  # b x 512 x 64 x 64
        out_conv5 = self.conv5_1(self.conv5(out_conv4))  # b x 512 x 32 x 32
        out_conv6 = self.conv6_1(self.conv6(out_conv5))  # b x 1024 x 16 x 16

        flow6       = self.predict_flow6(out_conv6)  # b x 2 x 16 x 16
        flow6_up    = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)  # b x 2 x 32 x 32
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)  # b x 512 x 32 x 32

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)  # b x 1026 x 32 x 32
        flow5       = self.predict_flow5(concat5)  # b x 2 x 32 x 32
        flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)  # b x 2 x 64 x 64
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)  # b x 770 x 64 x 64

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)  # b x 770 x 64 x 64
        flow4       = self.predict_flow4(concat4)  # b x 2 x 64 x 64
        flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3a)  # b x 2 x 128 x 128
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3a)  # b x 128 x 128 x 128

        concat3 = torch.cat((out_conv3a,out_deconv3,flow4_up),1)  # b x 386 x 128 x 128
        flow3       = self.predict_flow3(concat3)  # b x 2 x 128 x 128
        flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2a)  # b x 2 x 256 x 256
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2a)  # b x 64 x 256 x 256

        concat2 = torch.cat((out_conv2a,out_deconv2,flow3_up), 1)  # b x 194 x 256 x 256
        flow2 = self.predict_flow2(concat2)

        return flow2, flow3, flow4, flow5, flow6

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

