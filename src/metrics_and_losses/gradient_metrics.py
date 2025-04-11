import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_gradients(flow):   
    """ Compute the image gradients that are needed for the regularization"""
    gradient_ew = get_sobel_per_pixel_with_sqrt(flow[:, 0])
    gradient_ns = get_sobel_per_pixel_with_sqrt(flow[:, 1])
    gradient_flow = gradient_ew ** 2 + gradient_ns ** 2
    return torch.sqrt(gradient_flow)


class TotalVariationL1(nn.Module):
    def __init__(self):
        super(TotalVariationL1, self).__init__()

    def forward(self, prediction, edge_mask=None):
        """Compute total variation with edge mask separation.
        
        Args:
            prediction: Input tensor of shape (batch_size, channels, height, width)
            edge_mask: Binary mask indicating edge pixels
            maximum_displacements: Not used
            direction_id: Not used
            
        Returns:
            Tuple of (mean total variation, mean edge variation, mean non-edge variation)
        """
        _, _, h, w = prediction.shape
        
        # Compute total variation gradients
        gradient_image = get_total_variation_per_pixel(prediction)
        
        # Split into edge and non-edge regions
        edge_count = torch.sum(edge_mask)
        non_edge_count = torch.sum(1 - edge_mask)
        
        edge_gradient = gradient_image * edge_mask
        non_edge_gradient = gradient_image * (1 - edge_mask)

        # Compute normalized variations
        total_variation = torch.sum(gradient_image, dim=(-1, -2)) / (h * w)
        total_variation_edge = torch.sum(edge_gradient, dim=(-1, -2)) / max(edge_count, 1)
        total_variation_non_edge = torch.sum(non_edge_gradient, dim=(-1, -2)) / max(non_edge_count, 1)

        return total_variation.mean(), total_variation_edge.mean(), total_variation_non_edge.mean()


class L2Smoothness(nn.Module):
    def __init__(self):
        super(L2Smoothness, self).__init__()

    def forward(self, prediction, edge_mask=None):
        b, c, h, w = prediction.shape

        gradient_image = get_sobel_per_pixel(prediction)

        edge_count = torch.sum(edge_mask)
        non_edge_count = torch.sum(1 - edge_mask)
        edge_gradient = gradient_image * edge_mask
        non_edge_gradient = gradient_image * (1 - edge_mask)

        # Compute normalized variations
        total_variation = torch.sum(gradient_image, dim=(-1, -2)) / (h * w)
        total_variation_edge = torch.sum(edge_gradient, dim=(-1, -2)) / max(edge_count, 1)
        total_variation_non_edge = torch.sum(non_edge_gradient, dim=(-1, -2)) / max(non_edge_count, 1)

        return total_variation.mean(), total_variation_edge.mean(), total_variation_non_edge.mean()


#---------------------------
# Total variation filters
#--------------------------
def get_diff_tv(image):
    """Compute total variation differences in x / EW and y / NS directions. 
    Note: the outputs are not of the same shape as the input"""
    diff_x = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
    diff_y = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])
    return diff_x, diff_y


def get_total_variation_per_pixel(image):
    """Compute total variation per pixel by summing absolute differences in x and y directions.
    Note: the outputs are not of the same shape as the input
    """
    diff_x, diff_y = get_diff_tv(image)
    b, c, h, w = image.shape

    zeros_row = torch.zeros((b, c, h, 1), device=diff_x.device) 
    diff_x_extended = torch.cat((diff_x, zeros_row), dim=-1)

    zeros_col = torch.zeros((b, c, 1, w), device=diff_x.device)
    diff_y_extended = torch.cat((diff_y, zeros_col), dim=-2)
    
    return diff_x_extended + diff_y_extended


#---------------------------
# Sobel filters
#--------------------------
def get_sobel(image):
    """ Compute the Sobel gradient for each pixel in the x and y directions independently, for each channel independently.
    The input image is expected to have shape (batch_size, channels, height, width).
    The output is a tuple of two tensors, each with shape (batch_size, channels, height, width).
    """
    # Define Sobel kernels
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=image.device, dtype=image.dtype).view(1, 1, 3, 3)

    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], device=image.device, dtype=image.dtype).view(1, 1, 3, 3)

    # Expand filters to apply to each channel independently
    channels = image.shape[1]
    sobel_x = sobel_x.expand(channels, 1, 3, 3)
    sobel_y = sobel_y.expand(channels, 1, 3, 3)

    # Apply Sobel filters with groups to preserve channel independence, 
    # and remove border effects with padding=0 + pad the image with 1 pixel and constant padding=0
    # (the gradient should be 0 at the border)
    grad_x = F.conv2d(image, sobel_x, padding=0, groups=channels)
    grad_y = F.conv2d(image, sobel_y, padding=0, groups=channels)
    grad_x = F.pad(grad_x, (1, 1, 1, 1), mode='constant', value=0)
    grad_y = F.pad(grad_y, (1, 1, 1, 1), mode='constant', value=0)
    return grad_x, grad_y


def get_sobel_per_pixel(image):
    """ Compute the Sobel gradient for each pixel for each channel independently."""
    grad_x, grad_y = get_sobel(image)
    return torch.sqrt(grad_x ** 2 + grad_y ** 2)


def get_sobel_per_pixel_with_sqrt(tensor):
    # Define Sobel kernels
    tensor = tensor.view(tensor.shape[0], 1, tensor.shape[1], tensor.shape[2])
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=tensor.device, dtype=tensor.dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=tensor.device, dtype=tensor.dtype).view(1, 1, 3, 3)

    # Apply Sobel filter
    grad_x = F.conv2d(tensor, sobel_x, padding=1)
    grad_y = F.conv2d(tensor, sobel_y, padding=1)

    return torch.sqrt(grad_x ** 2 + grad_y ** 2) 
