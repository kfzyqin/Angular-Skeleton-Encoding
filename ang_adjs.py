import torch
import numpy as np

from graph.tools import normalize_adjacency_matrix


def get_ang_adjs(data_type):
    rtn_adjs = []

    if data_type == 'ntu':
        node_num = 25
        sym_pairs = ((21, 2), (11, 7), (18, 14), (20, 16), (24, 25), (22, 23))
        for a_sym_pair in sym_pairs:
            a_adj = np.eye(node_num)
            a_adj[:, a_sym_pair[0]-1] = 1
            a_adj[:, a_sym_pair[1]-1] = 1
            a_adj = torch.tensor(normalize_adjacency_matrix(a_adj))
            rtn_adjs.append(a_adj)

    return torch.cat(rtn_adjs, dim=0)
