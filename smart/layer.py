import torch
from torch import nn
from torch.nn import Parameter

from torch_geometric.nn import GATConv, GCNConv, GPSConv, GINConv, SAGEConv, LGConv, GATv2Conv,GCN2Conv,GraphConv


import torch.nn.functional as F

class SAGEConv_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv_Encoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, out_channels, normalize=True)
        self.conv2 = SAGEConv(out_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class SAGEConv_Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv_Decoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, in_channels, normalize=True)
        self.conv2 = SAGEConv(in_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

class GCNConv_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels, normalize=True)
        self.conv2 = GCNConv(out_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GCNConv_Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_Decoder, self).__init__()
        self.conv1 = GCNConv(in_channels, in_channels, normalize=True)
        self.conv2 = GCNConv(in_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

class GCN2Conv_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN2Conv_Encoder, self).__init__()
        self.conv1 = GCN2Conv(in_channels, out_channels, normalize=True)
        self.conv2 = GCN2Conv(out_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GCN2Conv_Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN2Conv_Decoder, self).__init__()
        self.conv1 = GCN2Conv(in_channels, in_channels, normalize=True)
        self.conv2 = GCN2Conv(in_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x
        
class GATConv_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATConv_Encoder, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels,heads=2,concat=False)
        self.conv2 = GATConv(out_channels, out_channels,heads=2,concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GATConv_Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATConv_Decoder, self).__init__()
        self.conv1 = GATConv(in_channels, in_channels,heads=2,concat=False)
        self.conv2 = GATConv(in_channels, out_channels,heads=2,concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x
        
class GraphConv_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConv_Encoder, self).__init__()
        self.conv1 = GraphConv(in_channels, out_channels)
        self.conv2 = GraphConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GraphConv_Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConv_Decoder, self).__init__()
        self.conv1 = GraphConv(in_channels, in_channels)
        self.conv2 = GraphConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

class SMART(torch.nn.Module):
    def __init__(self, hidden_dims,device,Conv_Encoder=SAGEConv_Encoder,Conv_Decoder=SAGEConv_Decoder):
        super(SMART, self).__init__()

        out_dim = hidden_dims[-1]
        self.encoders=[Conv_Encoder(in_dim, out_dim).to(device) for in_dim in hidden_dims[:-1]]
        self.fc=nn.Linear((len(hidden_dims)-1)*out_dim, out_dim)
        self.decoders=[Conv_Decoder(out_dim, in_dim).to(device) for in_dim in hidden_dims[:-1]]

    def forward(self, features, edge_indexs):
        x = [encoder(feature,edge_index) for encoder,feature,edge_index in zip(self.encoders,features,edge_indexs)]
        if len(x)==1:
            z=self.fc(x[0])
        else:
            z=self.fc(torch.cat(x, dim=1))
        x_rec = [decoder(z,edge_index) for decoder,feature,edge_index in zip(self.decoders,features,edge_indexs)]
        return z, x_rec