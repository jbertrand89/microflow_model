"""
This code was copied from https://github.com/princeton-vl/RAFT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.baselines.raft.update import BasicUpdateBlock, SmallUpdateBlock, BasicUpdateBlockNoCorr
from src.models.baselines.raft.extractor import BasicEncoder, SmallEncoder
from src.models.baselines.raft.corr import CorrBlock, AlternateCorrBlock
from src.models.baselines.raft.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        self.max_iterations = args.max_iterations

        if args.raft_small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.raft_small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.raft_dropout)
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.raft_dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.raft_dropout, input_channel=args.raft_input_channel)
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.raft_dropout, input_channel=args.raft_input_channel)
            # Added to remove the correlation layer from RAFT
            if args.model_name == "raft_no_corr":
                self.update_block = BasicUpdateBlockNoCorr(self.args, hidden_dim=hdim, input_dim=512)
            elif args.model_name == "raft":
                self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(
            self, x, args=None, flow_init=None, upsample=True, test_mode=False, save=False, crs_meta_data=None,
            transform_meta_data=None, frame_labels=None
    ):
        if self.args.model_name == "raft":
            return self.forward_raft(
                x, flow_init, upsample, test_mode, save, crs_meta_data, transform_meta_data, frame_labels)
        elif self.args.model_name == "raft_no_corr":  # Added to remove the correlation layer from RAFT
            return self.forward_raft_no_corr(
                x, flow_init, upsample, test_mode, save, crs_meta_data, transform_meta_data, frame_labels)

    def forward_raft(
            self, x, flow_init=None, upsample=True, test_mode=False, save=False, crs_meta_data=None,
            transform_meta_data=None, frame_labels=None):

        image1 = x[:, 0]
        image2 = x[:, 1]
        image1 = torch.tile(image1[..., None], (1, 1, 3))
        image2 = torch.tile(image2[..., None], (1, 1, 3))

        image1 = image1.permute(0, 3, 1, 2)
        image2 = image2.permute(0, 3, 1, 2)

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for _ in range(self.max_iterations):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions

    def forward_raft_no_corr(
            self, x, flow_init=None, upsample=True, test_mode=False, save=False, crs_meta_data=None,
            transform_meta_data=None, frame_labels=None):
        """ Added function for running RAFT without correlation
        """

        image1 = x[:, 0]
        image2 = x[:, 1]
        image1 = torch.tile(image1[..., None], (1, 1, 3))
        image2 = torch.tile(image2[..., None], (1, 1, 3))

        image1 = image1.permute(0, 3, 1, 2)
        image2 = image2.permute(0, 3, 1, 2)

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []

        fmap_concat = torch.cat([fmap1, fmap2], dim=1)  # b x 512 x 32 x 32
        for _ in range(self.max_iterations):
            coords1 = coords1.detach()
            # corr = corr_fn(coords1)  # index correlation volume
            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, fmap_concat, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
