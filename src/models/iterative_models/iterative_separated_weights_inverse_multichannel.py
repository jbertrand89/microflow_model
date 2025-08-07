import torch
import torch.nn as nn
import numpy as np

from src.models.model_utils.backbone_model_utils import load_backbone
from src.models.model_utils.iterative_model_utils import normalize_image, compute_regularization
from src.models.model_utils.warping_utils import warp_image_torch, get_inverse_flow
from src.models.model_utils.saving_utils import apply_convention_and_save


def get_iterative_separated_weights_inverse_multichannel_model(
        device, model_name, batch_norm, max_iterations, filenames, trained_end2end=False, args=None):
    model = IterativeSeparatedWeightsMultiChannel(device, model_name, batch_norm, max_iterations, args=args)
    if trained_end2end and filenames[0] is not None:
        model.load_state_dict(torch.load(filenames[0])['model_state_dict'])

    return model


class IterativeSeparatedWeightsMultiChannel(nn.Module):
    expansion = 1

    def __init__(self, device, model_name, batch_norm, max_iterations, args=None):
        super(IterativeSeparatedWeightsMultiChannel, self).__init__()
        self.max_iterations = max_iterations

        self.repeat_model = args.repeat_model
        self.repeat_first_iteration = args.repeat_first_iteration

        iteration_models = []
        for _ in range(max_iterations):
            iteration_models.append(load_backbone(model_name, batch_norm, device, args=args))
        self.iteration_models = nn.ModuleList(iteration_models)

        self.backbone_name = model_name

        # self.freeze_parameters_at_iteration(0)
        self.detach_gradients = args.detach_gradients
        self.renormalize = args.renormalize

    def forward(
            self, x, ptv=None, args=None, save=False, frame_labels=False, crs_meta_data=None, transform_meta_data=None,
            flow_gt=None, test_mode=False):

        image1 = x[:, 0]
        image2 = x[:, 1]
        post_not_norm = x[:, 2]
        image1 = torch.tile(image1[..., None], (1, 1, 3))
        image2 = torch.tile(image2[..., None], (1, 1, 3))
        post_not_norm = torch.tile(post_not_norm[..., None], (1, 1, 3))

        image1 = image1.permute(0, 3, 1, 2)  # 32, 3, 256, 256
        image2 = image2.permute(0, 3, 1, 2)  # 32, 3, 256, 256
        post_not_norm = post_not_norm.permute(0, 3, 1, 2)  # 32, 3, 256, 256
        N, _, H, W = image1.shape
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        post_not_norm = post_not_norm.contiguous()

        iteration_inputs, iteration_sum_optical_flows = [], []
        iteration_optical_flows, iteration_warped_images, iteration_warped_normalized_images = [], [], []

        for iteration in range(self.max_iterations):
            if iteration == 0:
                iteration_inputs.append(torch.cat([image1, image2], dim=1))
            else:
                iteration_inputs.append(torch.cat([image1, iteration_warped_normalized_images[iteration - 1]], dim=1))

            # predicts the residual flow
            current_iteration = 0 if self.repeat_first_iteration else iteration
            if self.backbone_name == "searaft":
                iteration_optical_flows.append(
                    self.iteration_models[current_iteration](iteration_inputs[iteration], args=args)[0][
                        -1])  # takes the last flow preficted
            else:
                iteration_optical_flows.append(self.iteration_models[current_iteration](iteration_inputs[iteration])[0])

            # sums residual flows to get the predicted flow
            if iteration == 0:
                iteration_sum_optical_flows.append(iteration_optical_flows[iteration])
            else:
                if self.detach_gradients:
                    iteration_sum_optical_flows[iteration - 1] = iteration_sum_optical_flows[iteration - 1].detach()
                iteration_sum_optical_flows.append(
                    iteration_sum_optical_flows[iteration - 1] + iteration_optical_flows[iteration])

            if self.renormalize:
                # warps the x2 image with the flow
                iteration_warped_images.append(warp_image_torch(post_not_norm, iteration_sum_optical_flows[iteration]))

                # renormalizes the image
                iteration_warped_normalized_images.append(normalize_image(iteration_warped_images[iteration]))
            else:
                iteration_warped_normalized_images.append(
                    warp_image_torch(image2, iteration_sum_optical_flows[iteration]))

        if save:
            for iteration in range(self.max_iterations):  # , 3, 4, 9]:

                reverse_prediction = get_inverse_flow(iteration_sum_optical_flows[iteration])
                apply_convention_and_save(
                    frame_labels,
                    reverse_prediction,
                    args.save_dir,
                    f"reverse_noisy_flow_iteration_{iteration}",
                    crs_meta_data,
                    transform_meta_data
                )

        if test_mode:
            for iteration in range(self.max_iterations):
                iteration_sum_optical_flows[iteration] = get_inverse_flow(iteration_sum_optical_flows[iteration])

        if args.regularization:
            for iteration in range(self.max_iterations):
                iteration_sum_optical_flows[iteration] = compute_regularization(iteration_sum_optical_flows[iteration],
                                                                                args, ptv)

        return iteration_sum_optical_flows

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

