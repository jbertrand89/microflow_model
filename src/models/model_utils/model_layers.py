import torch
import torch.nn as nn
import torch.nn.functional as F



def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


def conv_no_stride(batchNorm, in_planes, out_planes, kernel_size=3):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


def i_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bias = True):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
            nn.BatchNorm2d(out_planes),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
        )


def correlate(input1, input2):
    from spatial_correlation_sampler import spatial_correlation_sample
    out_corr = spatial_correlation_sample(
        input1,
        input2,
        kernel_size=1,
        patch_size=21,
        stride=1,
        padding=0,
        dilation_patch=2,
    )
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w) / input1.size(1)
    return F.leaky_relu_(out_corr, 0.1)


def predict_flow(in_planes, out_planes=2):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


def deconv(in_planes, out_planes, batchNorm=False):
    if batchNorm:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1,inplace=True)
        )

class Upsample(nn.Module):
    def __init__(
            self, in_channels, out_channels, with_conv=False, with_residual_conv_block=False, with_activation=False,
            bias=True, mode="nearest"):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=bias)
        self.with_residual_conv_block = with_residual_conv_block
        if self.with_residual_conv_block:
            self.residual_conv_block = None  # todo

        self.with_activation = with_activation
        if self.with_activation:
            self.activation = nn.LeakyReLU(0.1, inplace=True)  # todo change leakyReLU to ReLU

        self.mode = mode
        if self.mode == "shufflepixel":
            self.shuffle_pixel = torch.nn.PixelShuffle(2)

    def forward(self, x):
        if self.mode == "nearest":
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        elif self.mode == "bilinear":
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode=self.mode, align_corners=False, antialias=True)
        elif self.mode == "shufflepixel":
            x = self.shuffle_pixel(x)

        if self.with_residual_conv_block:
            x = self.residual_conv_block(x)

        if self.with_conv:
            x = self.conv(x)
        if self.with_activation:
            x = self.activation(x)
        return x


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation="leaky_relu"):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None
        self.activation = activation

    def forward(self, x):
        identity = x

        # Apply downsampling if specified
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)

        if self.activation.lower() == "relu":
            out = F.relu(out)
        elif self.activation.lower() == "leaky_relu":
            out = F.leaky_relu(out, negative_slope=0.1)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity

        if self.activation.lower() == "relu":
            out = F.relu(out)
        elif self.activation.lower() == "leaky_relu":
            out = F.leaky_relu(out, negative_slope=0.1)

        return out
