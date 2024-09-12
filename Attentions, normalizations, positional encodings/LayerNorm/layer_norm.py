import torch
import torch.nn as nn

'''
torch.nn.LayerNorm implements the operation as described in the paper
Layer Normalization https://arxiv.org/abs/1607.06450
2016 - Geoffrey E. Hinton and all. 
This is the "classic approach" which one and I used in class ClassicLayerNorm)
'''


class ClassicLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        """
        Custom Layer Normalization implementation.

        Parameters:
        - normalized_shape (int or tuple): Input shape from an expected input.
        - eps (float, optional): A value added to the denominator for numerical stability.
        """
        super(ClassicLayerNorm, self).__init__()
        self.eps = eps

        # Create learnable parameters for normalization.
        # These parameters will be learned during training.
        # and normalized_shape is the number of features in the tensor
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        """
        Forward pass of the layer.

        Parameters:
        - x (Tensor): Input tensor to be normalized.
        """
        # Calculate the mean and variance of the input.
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        # Normalize the input.
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)

        # Apply learnable parameters.
        return self.gamma * x_normalized + self.beta


'''
# Code from Google Gemma model HaggingFace repo
# https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma
# Article: Root Mean Square Layer Normalization
# https://arxiv.org/pdf/1910.07467.pdf

torch.rsqrt():
https://pytorch.org/docs/stable/generated/torch.rsqrt.html
'''


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * (1 + self.weight)


def main():
    # Example tensor (batch_size, num_features)
    # from uniform distribution on the interval [0,1)
    example_tensor = torch.rand(3, 4, requires_grad=False)  # 3 examples, 4 features each

    # Apply LayerNorm
    layer_norm1 = nn.LayerNorm(normalized_shape=4)
    layer_norm2 = ClassicLayerNorm(normalized_shape=4)
    layer_norm3 = GemmaRMSNorm(dim=4)
    torch_layer_norm = layer_norm1(example_tensor)
    classic_layer_norm = layer_norm2(example_tensor)
    gemma_rms_norm = layer_norm3(example_tensor)
    print(f'example_tensor = {example_tensor}')
    print(f'torch_layer_norm = {torch_layer_norm}')
    print(f'classic_layer_norm = {classic_layer_norm}')
    print(f'RMSNorm = {gemma_rms_norm}')


if __name__ == '__main__':
    main()
