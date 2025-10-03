import torch
import torch.nn as nn

class ZeroConv1d(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ZeroConv1d, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        
        # Initialize weights and biases to zero
        nn.init.constant_(self.linear.weight, 0)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    def zero_init(self):
        nn.init.constant_(self.linear.weight, 0)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        # x: [B, T, D]
        # Apply linear transformation along the last dimension
        # This will output a tensor of shape [B, T, out_features]
        x = self.linear(x)
        return x