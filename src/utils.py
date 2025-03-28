import os
import dgl
import torch
import shutil
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dgl.transforms import FeatMask


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_writable(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass

def evaluat_metrics(output, label):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    pre_y = (torch.sigmoid(output) > 0.5).numpy()
    truth_y = label.numpy()
    N, C = pre_y.shape

    for i in range(N):
        for j in range(C):
            if pre_y[i][j] == truth_y[i][j]:
                if truth_y[i][j] == 1:
                    TP += 1
                else:
                    TN += 1
            elif truth_y[i][j] == 1:
                FN += 1
            elif truth_y[i][j] == 0:
                FP += 1

        Precision = TP / (TP + FP + 1e-10)
        Recall = TP / (TP + FN + 1e-10)
        F1_score = 2 * Precision * Recall / (Precision + Recall + 1e-10)

    return F1_score

def get_bfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    candiate_node = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 20:
        random_node = random.randint(0, node_num - 1)
    candiate_node.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        cur_node = candiate_node.pop(0)
        selected_node.append(cur_node)

        for edge_index in node_to_edge_index[cur_node]:
            if edge_index not in selected_edge_index:
                selected_edge_index.append(edge_index)

                end_node = -1
                if ppi_list[edge_index][0] == cur_node:
                    end_node = ppi_list[edge_index][1]
                else:
                    end_node = ppi_list[edge_index][0]

                if end_node not in selected_node and end_node not in candiate_node:
                    candiate_node.append(end_node)
            else:
                continue

    return selected_edge_index


def get_dfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    stack = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 20:
        random_node = random.randint(0, node_num - 1)
    stack.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        cur_node = stack[-1]

        if cur_node in selected_node:
            flag = True

            for edge_index in node_to_edge_index[cur_node]:
                if flag:
                    end_node = -1

                    if ppi_list[edge_index][0] == cur_node:
                        end_node = ppi_list[edge_index][1]
                    else:
                        end_node = ppi_list[edge_index][0]

                    if end_node in selected_node:
                        continue
                    else:
                        stack.append(end_node)
                        flag = False
                else:
                    break

            if flag:
                stack.pop()
            continue

        else:
            selected_node.append(cur_node)

            for edge_index in node_to_edge_index[cur_node]:
                if edge_index not in selected_edge_index:
                    selected_edge_index.append(edge_index)

    return selected_edge_index


def random_feature_drop(node_features, drop_percent, device):

    drop_matrix = torch.bernoulli(
        torch.ones(node_features.shape, dtype=torch.float32, device=device),
        1 - drop_percent)

    droped_features = node_features * drop_matrix

    return droped_features

def random_feature_featmask(node_features, drop_percent, device):

    transform = FeatMask(p=drop_percent, node_feat_names=['feat'])

    g = dgl.graph((torch.tensor([]), torch.tensor([])), num_nodes=node_features.shape[0])
    g.ndata['feat'] = node_features

    g = transform(g)

    masked_features = g.ndata['feat']
    return masked_features


def random_edge_pert(edge_index, num_nodes, pert_percent, device):

    num_edges = edge_index[0].shape[0]

    pert_num_edges = int(num_edges * pert_percent)

    pert_idxs = np.random.choice(num_edges, pert_num_edges, replace=False)
    pert_idxs = torch.tensor(pert_idxs, dtype=torch.long, device=device)

    perturbed_edge_index = [
        edge_index[0].clone().to(device), edge_index[1].clone().to(device)
    ]

    perturbed_edge_index[0][pert_idxs] = torch.randint(0,
                                                       num_nodes,
                                                       (pert_num_edges, ),
                                                       device=device)
    perturbed_edge_index[1][pert_idxs] = torch.randint(0,
                                                       num_nodes,
                                                       (pert_num_edges, ),
                                                       device=device)

    return perturbed_edge_index


def random_edge_pert_g(edge_index, num_nodes, pert_percent, device):
    num_edges = edge_index[0].shape[0]
    pert_num_edges = int(num_edges * pert_percent)
    pert_idxs = np.random.choice(num_edges, pert_num_edges, replace=False)
    pert_idxs = torch.tensor(pert_idxs, dtype=torch.long, device=device)

    perturbed_edge_index = [
        edge_index[0].clone().to(device), edge_index[1].clone().to(device)
    ]
    
    perturbed_edge_index[0][pert_idxs] = torch.randint(0,
                                                       num_nodes,
                                                       (pert_num_edges, ),
                                                       device=device)
    perturbed_edge_index[1][pert_idxs] = torch.randint(0,
                                                       num_nodes,
                                                       (pert_num_edges, ),
                                                       device=device)

    perturbed_g = dgl.graph((perturbed_edge_index[0], perturbed_edge_index[1]), num_nodes=num_nodes)

    return perturbed_g


def get_cl_loss(h1, h2, margin):
    z1 = F.normalize(h1, dim=-1, p=2)
    z2 = F.normalize(h2, dim=-1, p=2)

    similarities = F.cosine_similarity(z1, z2, dim=-1)

    mask_positive = torch.eye(similarities.shape[0],
                              device=similarities.device).bool()
    mask_negative = ~mask_positive

    positive_logits = similarities * mask_positive

    negative_logits = torch.max(similarities * mask_negative, dim=0)[0]

    logits = positive_logits - negative_logits
    exp_logits = torch.exp(logits)
    denominator = exp_logits * mask_positive + torch.sum(
        exp_logits * mask_negative, dim=0)
    loss = -torch.log(exp_logits / denominator).mean()

    return loss

def compute_edge_type(batch_graph, device):
    edge_type_mapping = {item: index for index, item in enumerate(batch_graph.etypes)}
    edge_type = []
    for etype in batch_graph.etypes:
        edges = batch_graph.edges(etype=etype)
        num_edges = len(edges[0])
        edge_type.extend([edge_type_mapping[etype]] * num_edges)
    edge_type_tensor = torch.tensor(edge_type, device=device)

    return edge_type_tensor


def compute_node_type(batch_graph, device):
    node_type_mapping = {item: index for index, item in enumerate(batch_graph.ntypes)}
    node_type = []
    for ntype in batch_graph.ntypes:
        nodes = batch_graph.nodes(ntype=ntype)
        num_nodes = len(nodes)
        node_type.extend([node_type_mapping[ntype]] * num_nodes)
    node_type_tensor = torch.tensor(node_type, device=device)

    return node_type_tensor


def compute_edge_feat(param, batch_graph, device):
    edge_type_mapping = {item: index for index, item in enumerate(batch_graph.etypes)}
    edge_type = []
    for etype in batch_graph.etypes:
        edges = batch_graph.edges(etype=etype)
        num_edges = len(edges[0])
        edge_type.extend([edge_type_mapping[etype]] * num_edges)
    edge_type_tensor = torch.tensor(edge_type, device=device)
    edge_embedding = nn.Embedding(3, param['resid_hidden_dim'], device=device)
    edge_feat = edge_embedding(edge_type_tensor)

    return edge_feat

def compute_edge_feat(param, batch_graph, device): # 分
    edge_type_mapping = {item: index for index, item in enumerate(batch_graph.etypes)}
    edge_feat_dict = {}
    for etype in batch_graph.etypes:
        edges = batch_graph.edges(etype=etype)
        num_edges = len(edges[0])
        edge_type_tensor = torch.full((num_edges,), edge_type_mapping[etype], dtype=torch.long, device=batch_graph.device)
        edge_embedding = nn.Embedding(3, param['resid_hidden_dim'], device=device)
        edge_feat = edge_embedding(edge_type_tensor)
        edge_feat_dict[etype] = edge_feat
    
    return edge_feat_dict