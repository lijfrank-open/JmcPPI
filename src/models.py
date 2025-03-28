import dgl
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import HeteroGraphConv
from dgl.nn.pytorch.conv import GINConv

from module import GATDotConv
from utils import random_edge_pert_g, random_feature_drop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ProtIR(torch.nn.Module):

    def __init__(self, param):
        super(ProtIR, self).__init__()

        self.param = param
        self.num_layers = param['prot_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.layers = nn.ModuleList()

        self.layers.append(
            GINConv(nn.Sequential(
                nn.Linear(param['resid_hidden_dim'], param['prot_hidden_dim']),
                nn.ReLU(),
                nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']),
                nn.ReLU(), nn.BatchNorm1d(param['prot_hidden_dim'])),
                    aggregator_type='sum',
                    learn_eps=True))

        for i in range(self.num_layers - 1):
            self.layers.append(
                GINConv(nn.Sequential(
                    nn.Linear(param['prot_hidden_dim'],
                              param['prot_hidden_dim']), nn.ReLU(),
                    nn.Linear(param['prot_hidden_dim'],
                              param['prot_hidden_dim']), nn.ReLU(),
                    nn.BatchNorm1d(param['prot_hidden_dim'])),
                        aggregator_type='sum',
                        learn_eps=True))

        self.get_cl_loss = ContrastiveLoss(tau=param['temperature'])
        self.linear = nn.Linear(param['prot_hidden_dim'],
                                param['prot_hidden_dim'])
        self.fc = nn.Linear(param['prot_hidden_dim'] * 2, param['output_dim'])

    def forward(self, g, x, ppi_list, idx, trainW):

        if trainW:
            droped_x1 = random_feature_drop(x, self.param['pert_ratio'], device)
            
            perturbed_g1 = random_edge_pert_g(
                g.edges(),
                num_nodes=g.num_nodes(),
                pert_percent=self.param['pert_ratio'],
                device=device)

            droped_x2 = random_feature_drop(x, self.param['pert_ratio'],
                                            device)
            
            perturbed_g2 = random_edge_pert_g(
                g.edges(),
                num_nodes=g.num_nodes(),
                pert_percent=self.param['pert_ratio'],
                device=device)

            for l, layer in enumerate(self.layers):
                droped_x1 = layer(perturbed_g1, droped_x1)
                droped_x1 = self.dropout(droped_x1)
                droped_x2 = layer(perturbed_g2, droped_x2)
                droped_x2 = self.dropout(droped_x2)
                
                x = layer(g, x)
                x = self.dropout(x)

            cl_loss = self.get_cl_loss(x, droped_x1) + self.get_cl_loss(
                x, droped_x2)
        else:
            for l, layer in enumerate(self.layers):
                x = layer(g, x)
                x = self.dropout(x)
            cl_loss = 0
        
        x = self.dropout(F.relu(self.linear(x)))

        node_id = np.array(ppi_list)[idx]
        x1 = x[node_id[:, 0]]
        x2 = x[node_id[:, 1]]

        x = self.fc(torch.cat([torch.mul(x1, x2), x1 + x2],
                              dim=-1))

        return x, cl_loss


class ContrastiveLoss(torch.nn.Module): 

    def __init__(self, tau=0.5):
        super(ContrastiveLoss, self).__init__()
        self.tau = tau 

    def forward(self, features1, features2):

        features1 = F.normalize(features1, p=2, dim=-1)
        features2 = F.normalize(features2, p=2, dim=-1)

        sim = torch.div(features1 @ features2.t(), self.tau) 
        pos_mask = torch.eye(features1.size(0), 
                             device=features1.device,
                             dtype=torch.bool)
        neg_mask = ~pos_mask 
        exp_sim = torch.exp(sim) 

        exp_pos_logits = torch.diag(exp_sim * pos_mask)
        denominator = torch.sum(exp_sim * neg_mask, dim=0)

        loss = -torch.log(exp_pos_logits / denominator).mean() 

        return loss


class ResiSC_Encoder(nn.Module):

    def __init__(self, param, data_loader):
        super(ResiSC_Encoder, self).__init__()

        self.data_loader = data_loader
        self.num_layers = param['resid_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.gnnlayers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.param = param

        self.fc_dim = nn.Linear(param['input_dim'], param['resid_hidden_dim'])

        for i in range(self.num_layers):
            self.gnnlayers.append(HeteroGraphConv({'SEQ' : GATDotConv(param['resid_hidden_dim'], param['resid_hidden_dim'], param['num_heads'], param['dropout_ratio'], allow_zero_in_degree=True), #
                                            'STR_KNN' : GATDotConv(param['resid_hidden_dim'], param['resid_hidden_dim'], param['num_heads'], param['dropout_ratio'], allow_zero_in_degree=True), 
                                            'STR_DIS' : GATDotConv(param['resid_hidden_dim'], param['resid_hidden_dim'], param['num_heads'], param['dropout_ratio'], allow_zero_in_degree=True)}, aggregate='sum'))
            self.fcs.append(nn.Linear(param['resid_hidden_dim'], param['resid_hidden_dim']))
            self.norms.append(nn.BatchNorm1d(param['resid_hidden_dim']))

    def forward(self):

        prot_embed_list = []

        for iter, batch_graph in enumerate(
                self.data_loader):

            batch_graph.to(device)
            x = batch_graph.ndata['x']

            batch_graph.ndata['h'] = self.encoding(batch_graph, x)

            prot_embed = dgl.mean_nodes(batch_graph,
                                        'h').detach().cpu()
            prot_embed_list.append(prot_embed)

        return torch.cat(prot_embed_list, dim=0)

    def encoding(self, batch_graph, x):

        x = self.fc_dim(x)

        for l, layer in enumerate(self.gnnlayers):
            x = torch.mean(layer(batch_graph, {'amino_acid': x})['amino_acid'], dim=1)
            x = self.norms[l](F.relu(self.fcs[l](x)))
            if l != self.num_layers - 1:
                x = self.dropout(x)

        return x


class ResiSC_Decoder(nn.Module):

    def __init__(self, param):
        super(ResiSC_Decoder, self).__init__()

        self.num_layers = param['resid_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.gnnlayers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fcs = nn.ModuleList()

        for i in range(self.num_layers):
            self.gnnlayers.append(HeteroGraphConv({'SEQ' : GATDotConv(param['resid_hidden_dim'], param['resid_hidden_dim'], param['num_heads'], param['dropout_ratio'], allow_zero_in_degree=True), #
                                            'STR_KNN' : GATDotConv(param['resid_hidden_dim'], param['resid_hidden_dim'], param['num_heads'], param['dropout_ratio'], allow_zero_in_degree=True), 
                                            'STR_DIS' : GATDotConv(param['resid_hidden_dim'], param['resid_hidden_dim'], param['num_heads'], param['dropout_ratio'], allow_zero_in_degree=True)}, aggregate='sum'))
            self.fcs.append(nn.Linear(param['resid_hidden_dim'], param['resid_hidden_dim']))
            self.norms.append(nn.BatchNorm1d(param['resid_hidden_dim']))

        self.fc_dim = nn.Linear(param['resid_hidden_dim'], param['input_dim'])

    def decoding(self, batch_graph, x):

        for l, layer in enumerate(self.gnnlayers):

            x = torch.mean(layer(batch_graph, {'amino_acid': x})['amino_acid'], dim=1)
            x = self.norms[l](F.relu(self.fcs[l](x)))
            if l != self.num_layers:
                x = self.dropout(x)

        x = self.fc_dim(x)

        return x

class RecNet(nn.Module): 
    def __init__(self, param, data_loader):
        super(RecNet, self).__init__()

        self.param = param

        self.Encoder = ResiSC_Encoder(param, data_loader)
        self.Decoder = ResiSC_Decoder(param)

        self.edge_embedding = nn.Embedding(3, param['resid_hidden_dim'])

    def forward(self, batch_graph):
        x = batch_graph.ndata['x']

        z = self.Encoder.encoding(batch_graph, x)
        recon_x = self.Decoder.decoding(batch_graph, z)

        recon_loss = F.mse_loss(recon_x, batch_graph.ndata['x'])

        mask_x = batch_graph.ndata['x'].clone()
        num_masked_rows = int(self.param['rec_mask_ratio'] * mask_x.shape[0])
        mask_index = torch.randperm(mask_x.shape[0])[:num_masked_rows]
        mask_x[mask_index] = 0.0
        mask_z = self.Encoder.encoding(batch_graph, mask_x)
        mask_recon_x = self.Decoder.decoding(batch_graph, mask_z)

        x = F.normalize(mask_recon_x[mask_index], p=2, dim=-1, eps=1e-12)
        y = F.normalize(batch_graph.ndata['x'][mask_index],
                        p=2,
                        dim=-1,
                        eps=1e-12)
        mask_recon_loss = ((1 - (x * y).sum(dim=-1)).pow_(
            self.param['sce_scale']))

        return z, recon_loss, mask_recon_loss.sum() / (
            mask_recon_loss.shape[0] + 1e-12)