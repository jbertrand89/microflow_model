import torch.nn as nn


class IntermediateL1Loss(nn.Module):
    def __init__(self, gamma=1):
        super(IntermediateL1Loss, self).__init__()
        self.gamma = gamma

    def forward(self, predictions, target):
        n_predictions = len(predictions)
        flow_loss = 0.0

        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            i_loss = (predictions[i] - target).abs()
            flow_loss += i_weight * i_loss.mean()
        return flow_loss
