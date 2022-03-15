import sys
sys.path.extend(['../'])

import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, TensorDataset
from numpy import inf

import scipy.fftpack

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True,
                 tgt_labels=None, frame_len=300, **kwargs):
        """
        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.tgt_labels = tgt_labels
        self.kwargs = kwargs

        # other parameters
        self.frame_len = frame_len

        self.load_data()

        # Internal dataloader parameters
        # self.load_bch_sz = 2000
        # self.internal_dataloader = self.get_a_dataloader()

        if normalization:
            self.get_mean_map()

    def get_a_dataloader(self):
        a_dataset = TensorDataset(torch.tensor(self.data))
        a_dataloader = torch.utils.data.DataLoader(
            dataset=a_dataset,
            batch_size=self.load_bch_sz,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )
        return a_dataloader

    def load_data(self):
        # data: N C T V M
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        # self.label = np.array(self.label)

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # Use tgt labels
        if self.tgt_labels is not None:
            self.label = np.array(self.label)
            tmp_data = None
            tmp_label = None
            for a_tgt_label in self.tgt_labels:
                selected_idxes = np.array(self.label) == a_tgt_label
                if tmp_data is None:
                    tmp_data = self.data[selected_idxes]
                    tmp_label = self.label[selected_idxes]
                else:
                    tmp_data = np.concatenate((tmp_data, self.data[selected_idxes]), axis=0)
                    tmp_label = np.concatenate((tmp_label, self.label[selected_idxes]), axis=0)
            self.data = tmp_data
            self.label = tmp_label

        if 'process_type' in self.kwargs:
            self.process_data(process_type=self.kwargs['process_type'])

        if self.debug:
            self.label = self.label[0:1000]
            self.data = self.data[0:1000]
            self.sample_name = self.sample_name[0:1000]
            # debug_tgt = 4206  # 4206  #
            # hard_samples = [9566, 13297, 15239, 13351, 11670, 8935, 2815, 9329, 15238, 8896]
            # self.label = list(self.label[i] for i in hard_samples)
            # self.data = list(self.data[i] for i in hard_samples)
            # self.sample_name = list(self.sample_name[i] for i in hard_samples)

        # Discrete cosine transform
        if 'dct' in self.kwargs:
            self.dct_data(self.kwargs['dct'])
            print('Discrete cosine transform completed. DCT type: ', self.kwargs['dct'])

    def dct_data(self, dct_op):
        dct_out = scipy.fftpack.dct(self.data, axis=2)
        if dct_op == 'overwrite':
            self.data = dct_out
        elif dct_op == 'concat':
            self.data = np.concatenate((self.data, dct_out), axis=1)
        elif dct_op == 'lengthen':
            dct_out = dct_out[:, :, :(self.frame_len // 2), :, :]
            self.data = np.concatenate((self.data, dct_out), axis=2)

    def process_data(self, process_type):
        rtn_data = []
        rtn_label = []
        if process_type == 'single_person':
            data_idx = 0
            for a_data in self.data:
                for ppl_id in range(self.data.shape[-1]):
                    a_ppl = a_data[:, :, :, ppl_id]
                    # comment the below one
                    # if np.max(a_ppl) > 0.01:

                    # Keep all data (some mutual data also contain zero)
                    if np.max(a_ppl) > -1:
                        rtn_data.append(np.expand_dims(a_ppl, axis=-1))
                        rtn_label.append(self.label[data_idx])
                data_idx += 1
        else:
            raise NotImplementedError
        rtn_data = np.stack(rtn_data, axis=0)
        rtn_label = np.stack(rtn_label, axis=0)

        # relabel data to consider actor and receiver
        rtn_label = self.relabel_by_energy()

        self.data = rtn_data
        self.label = rtn_label

    def get_energy(self, s):  # ctv
        index = s.sum(-1).sum(0) != 0  # select valid frames
        s = s[:, index, :]
        if len(s) != 0:
            s = s[0, :, :].std() + s[1, :, :].std() + s[2, :, :].std()  # three channels
        else:
            s = 0
        return s

    def relabel_by_energy(self):
        rtn_label = []
        for a_idx, a_data in enumerate(self.data):
            person_1 = a_data[:, :, :, 0]  # C,T,V
            person_2 = a_data[:, :, :, 1]  # C,T,V
            energy_1 = self.get_energy(person_1)
            energy_2 = self.get_energy(person_2)
            if energy_1 > energy_2:  # first kicks the second
                rtn_label.append((1, 0))
            else:
                rtn_label.append((0, 1))
        return np.concatenate(rtn_label, axis=0)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # Remove NAN
        data_numpy = np.nan_to_num(data_numpy)
        data_numpy[data_numpy == -inf] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        # 如果label只有一个值的话
        the_label = np.array(self.label)
        if len(the_label.shape) == 1 or the_label.shape[1] == 1:
            rank = score.argsort()
            hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
            return sum(hit_top_k) * 1.0 / len(hit_top_k)
        # label里面包含fine grain的label
        else:
            act_label = the_label[:, 0]
            fgr_label = the_label[:, 1]
            act_score, fgr_score = score
            rank_act = act_score.argsort()
            rank_fgr = fgr_score.argsort()
            hit_top_k_acc = [l in rank_act[i, -top_k:] for i, l in enumerate(act_label)]
            hit_top_k_fgr = [l in rank_fgr[i, -top_k:] for i, l in enumerate(fgr_label)]
            return sum(hit_top_k_acc) * 1.0 / len(hit_top_k_acc), \
                   sum(hit_top_k_fgr) * 1.0 / len(hit_top_k_fgr)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path:
    :param label_path:
    :param vid: the id of sample
    :param graph:
    :param is_3d: when vis NTU, set it True
    :return:
    '''
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward
            pose = []
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')
                plt.pause(0.01)


if __name__ == '__main__':
    import os
    os.environ['DISPLAY'] = 'localhost:10.0'
    data_path = "../data/ntu/xview/val_data_joint.npy"
    label_path = "../data/ntu/xview/val_label.pkl"
    graph = 'graph.ntu_rgb_d.Graph'
    test(data_path, label_path, vid='S004C001P003R001A032', graph=graph, is_3d=True)
    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)
