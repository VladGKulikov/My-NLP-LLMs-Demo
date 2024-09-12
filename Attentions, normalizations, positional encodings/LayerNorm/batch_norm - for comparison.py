import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Running mean and variance
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # Normalize and scale
            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

            # Update running stats
            self.running_mean = (self.momentum * batch_mean
                                 + (1 - self.momentum) * self.running_mean)
            self.running_var = (self.momentum * batch_var
                                + (1 - self.momentum) * self.running_var)
        else:
            # Normalize using running estimates
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        # Scale and shift
        x_norm = self.gamma * x_norm + self.beta
        return x_norm


