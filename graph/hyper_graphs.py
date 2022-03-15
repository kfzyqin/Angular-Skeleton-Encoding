import sys
import numpy as np
from graph import tools
import torch

sys.path.insert(0, '')
sys.path.extend(['../'])


local_bone_hyper_edge_dict = {
    'ntu': (
        (1, 17, 13), (2, 21, 1), (3, 4, 21), (4, 4, 4), (5, 21, 6), (6, 5, 7), (7, 6, 8), (8, 23, 22),
        (9, 10, 21), (10, 11, 9), (11, 12, 10), (12, 24, 25), (13, 1, 14), (14, 13, 15), (15, 14, 16),
        (16, 16, 16), (17, 18, 1), (18, 19, 17), (19, 20, 18), (20, 20, 20), (21, 9, 5), (22, 8, 23),
        (23, 8, 22), (24, 25, 12), (25, 24, 12)
    )
}


center_hyper_edge_dict = {
    'ntu': (
        (1, 2, 21), (2, 2, 21), (3, 2, 21), (4, 2, 21), (5, 2, 21), (6, 2, 21), (7, 2, 21), (8, 2, 21),
        (9, 2, 21), (10, 2, 21), (11, 2, 21), (12, 2, 21), (13, 2, 21), (14, 2, 21), (15, 2, 21),
        (16, 2, 21), (17, 2, 21), (18, 2, 21), (19, 2, 21), (20, 2, 21), (21, 2, 21), (22, 2, 21),
        (23, 2, 21), (24, 2, 21), (25, 2, 21),
    )
}


figure_l_hyper_edge_dict = {
    'ntu': (
        (1, 24, 25), (2, 24, 25), (3, 24, 25), (4, 24, 25), (5, 24, 25), (6, 24, 25), (7, 24, 25), (8, 24, 25),
        (9, 24, 25), (10, 24, 25), (11, 24, 25), (12, 24, 25), (13, 24, 25), (14, 24, 25), (15, 24, 25),
        (16, 24, 25), (17, 24, 25), (18, 24, 25), (19, 24, 25), (20, 24, 25), (21, 24, 25), (22, 24, 25),
        (23, 24, 25), (24, 24, 25), (25, 24, 25),
    )
}


figure_r_hyper_edge_dict = {
    'ntu': (
        (1, 22, 23), (2, 22, 23), (3, 22, 23), (4, 22, 23), (5, 22, 23), (6, 22, 23), (7, 22, 23), (8, 22, 23),
        (9, 22, 23), (10, 22, 23), (11, 22, 23), (12, 22, 23), (13, 22, 23), (14, 22, 23), (15, 22, 23),
        (16, 22, 23), (17, 22, 23), (18, 22, 23), (19, 22, 23), (20, 22, 23), (21, 22, 23), (22, 22, 23),
        (23, 22, 23), (24, 22, 23), (25, 22, 23),
    )
}


hand_hyper_edge_dict = {
    'ntu': (
        (1, 24, 22), (2, 24, 22), (3, 24, 22), (4, 24, 22), (5, 24, 22), (6, 24, 22), (7, 24, 22), (8, 24, 22),
        (9, 24, 22), (10, 24, 22), (11, 24, 22), (12, 24, 22), (13, 24, 22), (14, 24, 22), (15, 24, 22),
        (16, 24, 22), (17, 24, 22), (18, 24, 22), (19, 24, 22), (20, 24, 22), (21, 24, 22), (22, 24, 22),
        (23, 24, 22), (24, 24, 22), (25, 24, 22),
    )
}


elbow_hyper_edge = {
    'ntu': (
        (1, 10, 6), (2, 10, 6), (3, 10, 6), (4, 10, 6), (5, 10, 6), (6, 10, 6), (7, 10, 6), (8, 10, 6),
        (9, 10, 6), (10, 10, 6), (11, 10, 6), (12, 10, 6), (13, 10, 6), (14, 10, 6), (15, 10, 6),
        (16, 10, 6), (17, 10, 6), (18, 10, 6), (19, 10, 6), (20, 10, 6), (21, 10, 6), (22, 10, 6),
        (23, 10, 6), (24, 10, 6), (25, 10, 6),
    )
}


foot_hyper_edge = {
    'ntu': (
        (1, 20, 16), (2, 20, 16), (3, 20, 16), (4, 20, 16), (5, 20, 16), (6, 20, 16), (7, 20, 16), (8, 20, 16),
        (9, 20, 16), (10, 20, 16), (11, 20, 16), (12, 20, 16), (13, 20, 16), (14, 20, 16), (15, 20, 16),
        (16, 20, 16), (17, 20, 16), (18, 20, 16), (19, 20, 16), (20, 20, 16), (21, 20, 16), (22, 20, 16),
        (23, 20, 16), (24, 20, 16), (25, 20, 16),
    )
}


def get_hyper_edge(dataset, edge_type):
    if edge_type == 'local_bone':
        tgt_dict = local_bone_hyper_edge_dict
    elif edge_type == 'center':
        tgt_dict = center_hyper_edge_dict
    elif edge_type == 'figure_l':
        tgt_dict = figure_l_hyper_edge_dict
    elif edge_type == 'figure_r':
        tgt_dict = figure_r_hyper_edge_dict
    elif edge_type == 'hand':
        tgt_dict = hand_hyper_edge_dict
    elif edge_type == 'elbow':
        tgt_dict = elbow_hyper_edge
    elif edge_type == 'foot':
        tgt_dict = foot_hyper_edge
    else:
        raise NotImplementedError
    tgt_hyper_edge = tgt_dict[dataset]
    if 'ntu' in dataset:
        node_num = 25
        hyper_edge_adj = torch.zeros((node_num, node_num))
    else:
        raise NotImplementedError
    for i in range(node_num):
        edge_idx = 0
        for a_hyper_edge in tgt_hyper_edge:
            if (i+1) in a_hyper_edge:
                hyper_edge_adj[i][edge_idx] = 1
            edge_idx += 1

    return hyper_edge_adj



def get_local_bone_hyper_adj(dataset):
    if 'ntu' in dataset:
        the_hyper_edge = [
            (17, 13),
            (21, 1),
            (4, 21),
            (21, 6),
            (5, 7),
            (6, 8),
            (23, 22),
            (10, 21),
            (11, 9),
            (12, 10),
            (24, 25),
            (1, 14),
            (13, 15),
            (14, 16),
            (18, 1),
            (19, 17),
            (20, 18),
            (20, 20),
            (9, 5),
            (8, 23),
            (8, 22),
            (25, 12),
            (24, 12)
        ]
    else:
        raise NotImplementedError
    return np.array(the_hyper_edge)


def get_ntu_local_bone_neighbor():
    local_bone_inward_ori_index = [(25, 24), (25, 12), (24, 25), (24, 12), (12, 24), (12, 25),
                                   (11, 12), (11, 10), (10, 11), (10, 9), (9, 10), (9, 21),
                                   (21, 9), (21, 5), (5, 21), (5, 6), (6, 5), (6, 7), (7, 6), (7, 8),
                                   (8, 23), (8, 22), (22, 8), (22, 23), (23, 8), (23, 22), (3, 4),
                                   (3, 21), (2, 21), (2, 1), (1, 17), (1, 13), (17, 18), (17, 1),
                                   (18, 19), (18, 17), (19, 20), (19, 18), (13, 1), (13, 14),
                                   (14, 13), (14, 15), (15, 14), (15, 16)
                                   ]
    inward = [(i - 1, j - 1) for (i, j) in local_bone_inward_ori_index]
    outward = [(j, i) for (i, j) in inward]
    neighbor = inward + outward
    return neighbor


class LocalBoneAdj:
    def __init__(self, dataset):
        if 'ntu' in dataset:
            num_node = 25
            self.edges = get_ntu_local_bone_neighbor()
        elif 'kinetics' in dataset:
            num_node = 18
            raise NotImplementedError
        else:
            raise NotImplementedError
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)


if __name__ == '__main__':
    rtn = get_local_bone_hyper_edges('ntu')
    print('rtn: ', rtn)
