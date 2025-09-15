import torch
import torch.nn as nn

from src.models.model_utils.backbone_model_utils import load_backbone
from src.models.model_utils.iterative_model_utils import normalize_image, compute_regularization
from src.models.model_utils.warping_utils import warp_image_torch, get_inverse_flow
from src.models.model_utils.saving_utils import apply_convention_and_save


def get_iterative_separated_weights_inverse_model(
        device, model_name, batch_norm, max_iterations, filenames, trained_end2end=False, args=None):
    model = IterativeSeparatedWeightsInverse(device, model_name, batch_norm, max_iterations, args=args)
    if trained_end2end and filenames[0] is not None:
        model.load_state_dict(torch.load(filenames[0])['model_state_dict'])

    return model


class IterativeSeparatedWeightsInverse(nn.Module):
    expansion = 1

    def __init__(self, device, model_name, batch_norm, max_iterations, args=None):
        super(IterativeSeparatedWeightsInverse, self).__init__()
        self.max_iterations = max_iterations

        iteration_models = []
        for _ in range(max_iterations):
            iteration_models.append(load_backbone(model_name, batch_norm, device))
        self.iteration_models = nn.ModuleList(iteration_models)
        self.backbone_name = model_name

        self.args = args

    def forward(
            self, x, ptv=None, args=None, save=False, frame_labels=False, crs_meta_data=None, transform_meta_data=None,
            flow_gt=None, test_mode=False):

        b, _, h, w = x.shape

        image2 = x[:, 1].view(b, 1, h, w)

        iteration_inputs, iteration_sum_optical_flows, noisy_iteration_sum_optical_flows = [], [], []
        iteration_optical_flows, iteration_warped_images, iteration_warped_normalized_images = [], [], []

        for iteration in range(self.max_iterations):
            # inputs for the iterations
            iteration_inputs.append(x[:, 0:2].clone())
            if iteration > 0:
                iteration_inputs[iteration][:, 1] = iteration_warped_normalized_images[iteration - 1][:, 0]

            # predicts the residual flow
            iteration_optical_flows.append(self.iteration_models[iteration](iteration_inputs[iteration])[0])

            # sums residual flows to get the predicted flow
            if iteration == 0:
                iteration_sum_optical_flows.append(iteration_optical_flows[iteration])
            else:
                iteration_sum_optical_flows[iteration - 1] = iteration_sum_optical_flows[iteration - 1].detach()
                iteration_sum_optical_flows.append(iteration_sum_optical_flows[iteration - 1] + iteration_optical_flows[iteration])

            interpolated_flow = torch.nn.functional.interpolate(
                iteration_sum_optical_flows[iteration], (h, w), mode='bilinear', align_corners=False)

            iteration_warped_normalized_images.append(
                warp_image_torch(
                    image2,
                    interpolated_flow
                ))

        if args.regularization:
            iteration_sum_optical_flows[-1] = compute_regularization(iteration_sum_optical_flows[-1], args, ptv)

        if save:
            for iteration in range(self.max_iterations):
                reverse_prediction = get_inverse_flow(iteration_sum_optical_flows[iteration])
                apply_convention_and_save(
                    frame_labels,
                    reverse_prediction,
                    args.save_dir,
                    f"reverse_flow_iteration_{iteration}",
                    crs_meta_data,
                    transform_meta_data
                )

            reverse_prediction = get_inverse_flow(iteration_sum_optical_flows[-1])
            apply_convention_and_save(
                frame_labels,
                reverse_prediction,
                args.save_dir,
                f"reverse_flow_iteration_final",
                crs_meta_data,
                transform_meta_data
            )

        if test_mode:
            for iteration in range(self.max_iterations):
                iteration_sum_optical_flows[iteration] = get_inverse_flow(iteration_sum_optical_flows[iteration])

        return iteration_sum_optical_flows


    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]


    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

