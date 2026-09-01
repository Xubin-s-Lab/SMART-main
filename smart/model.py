import torch
from torch import nn
from torch.nn import Parameter
from smart.layer import SAGEConv_Encoder,SAGEConv_Decoder, GCNConv_Encoder, GCNConv_Decoder, GCN2Conv_Encoder, GCN2Conv_Decoder, GATConv_Encoder, GATConv_Decoder, GraphConv_Encoder, GraphConv_Decoder
import torch.nn.functional as F

class SMART(torch.nn.Module):
    """
    SMART: A modular multi-modal graph representation learning model.

    This model uses an encoder-decoder architecture for each modality,
    projects the learned embeddings into a shared latent space, and reconstructs
    input features for self-supervised training.

    Parameters
    ----------
    hidden_dims : list of int
        List specifying input dimensions of each modality and shared hidden dimension.
        Example: [in_dim_mod1, in_dim_mod2, ..., latent_dim].
    device : torch.device
        Device to place model modules on.
    Conv_Encoder : class
        Encoder architecture (default: SAGEConv_Encoder).
    Conv_Decoder : class
        Decoder architecture (default: SAGEConv_Decoder).
    """
    def __init__(self, hidden_dims, device, Conv_Encoder=SAGEConv_Encoder, Conv_Decoder=SAGEConv_Decoder):
        super(SMART, self).__init__()
        out_dim = hidden_dims[-1]

        # One encoder per modality
        self.encoders = nn.ModuleList([Conv_Encoder(in_dim, out_dim).to(device) for in_dim in hidden_dims[:-1]])
        self.fc = nn.Linear((len(hidden_dims) - 1) * out_dim, out_dim)

        # One decoder per modality
        self.decoders = nn.ModuleList([Conv_Decoder(out_dim, in_dim).to(device) for in_dim in hidden_dims[:-1]])

    def forward(self, features, edge_indexs):
        """
        Forward pass of the SMART model.

        Parameters
        ----------
        features : list of torch.Tensor
            Node features for each modality. Each tensor shape: [num_nodes, in_dim_mod].
        edge_indexs : list of torch.LongTensor
            Graph connectivity for each modality. Each tensor shape: [2, num_edges].

        Returns
        -------
        z : torch.Tensor
            Latent shared representation of shape [num_nodes, latent_dim].
        x_rec : list of torch.Tensor
            Reconstructed features for each modality.
        """
        # Encode each modality
        x = [encoder(feature, edge_index) for encoder, feature, edge_index in zip(self.encoders, features, edge_indexs)]

        # Concatenate or directly project
        if len(x) == 1:
            z = self.fc(x[0])
        else:
            z = self.fc(torch.cat(x, dim=1))

        # Decode each modality
        x_rec = [decoder(z, edge_index) for decoder, feature, edge_index in zip(self.decoders, features, edge_indexs)]
        return z, x_rec
