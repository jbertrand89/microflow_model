import torch
import torch.nn.functional as F


def EPE(input_flow, target_flow, sparse=False):
    error_map = target_flow - input_flow
    EPE_2_norm = torch.norm(error_map, p=2, dim=1).mean()
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)
        EPE_map = EPE_map[~mask]

    return EPE_2_norm


def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
    return EPE(upsampled_output, target, sparse)
