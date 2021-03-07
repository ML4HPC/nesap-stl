"""PredRNN++ model specification in PyTorch.

Adapted from https://github.com/ML4HPC/predrnn-pp
"""

# Externals
import torch

# Locals
from .layers import CausalLSTMStack

class PredRNNPP(torch.nn.Module):
    """The PredRNN++ model"""

    def __init__(self, filter_size=3, num_dims=2,
                 num_hidden=[128, 64, 64, 64, 16]):
        super().__init__()

        self.clstm = CausalLSTMStack(filter_size=filter_size,
                                     num_dims=num_dims,
                                     channels=num_hidden)
        if num_dims == 2:
            self.decoder = torch.nn.Conv2d(num_hidden[-1], num_hidden[-1], 1)
        elif num_dims == 3:
            self.decoder = torch.nn.Conv3d(num_hidden[-1], num_hidden[-1], 1)
        else:
            raise ValueError(f'num_dims value {num_dims} not allowed')

    def forward(self, x, hidden_states=None):
        # Initialize hidden states
        if hidden_states is None:
            h, c, m, z = [None]*4
        else:
            assert len(hidden_states) == 4
            h, c, m, z = hidden_states
        outputs = []

        # Loop over the sequence
        for t in range(x.shape[1]):
            h, c, m, z = self.clstm(x[:,t], h, c, m, z)
            outputs.append(self.decoder(h[-1])) #.permute(0, -1, 1, 2)

        # Stack outputs along time axis, last hidden states
        return torch.stack(outputs, dim=1), (h, c, m, z)


class RNNClassification(torch.nn.Module):
    """
    for classification
    """
    def __init__(self, filter_size, num_dims,
                 num_hidden, num_classes):
        super().__init__()
        self.encoder = PredRNNPP(filter_size, num_dims, num_hidden)
        self.classification_head = toch.nn.Linear(128, num_classes)  # 128 is hard-coded.
        # Must be modified to be alligned with hidden state size of PredRNNPP

    def foward(self, x):
        stacks, hiddens = self.encoder(x)
        return self.classification_head(hiddens[0])


def build_model(**kwargs):
    return PredRNNPP(**kwargs)
