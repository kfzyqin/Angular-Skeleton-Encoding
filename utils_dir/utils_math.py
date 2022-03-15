import math
import torch


class Embedder_DCT:
    def __init__(self, frm_len, multires, inc_input=True, inc_func='linear'):
        self.frm_len = frm_len
        self.multires = multires
        self.inc_input = inc_input
        self.inc_func = inc_func
        self.periodic_fns = [torch.cos]

        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        if self.inc_input:
            embed_fns.append(lambda x, y: x)  # with x

        N_freqs = self.multires

        freq_bands = []
        for k in range(1, N_freqs+1):
            if self.inc_func == 'linear':
                a_freq = k
            elif self.inc_func == 'exp':
                a_freq = 2 ** (k-1)
            elif self.inc_func == 'pow':
                a_freq = k ** 2
            else:
                raise NotImplementedError('Unsupported inc_func.')

            freq_bands.append(math.pi / self.frm_len * a_freq)  # This is DCT

        freq_bands = torch.tensor(freq_bands)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, frm_idx, p_fn=p_fn, freq=freq: (x * p_fn(freq * (frm_idx + 1/2))))  # this is DCT

        self.embed_fns = embed_fns

    def embed(self, inputs, dim):
        t_len_all = inputs.shape[2]
        time_list = []
        for t_idx in range(t_len_all):
            a_series = inputs[:, :, t_idx, :, :].unsqueeze(2)
            # new_time_list = torch.cat([fn(a_series, t_idx) for fn in self.embed_fns], dim)  # DCT

            # To try positional encoding
            new_time_list = []
            for fn in self.embed_fns:
                a_new_one = fn(a_series, t_idx)
                new_time_list.append(a_new_one)
            new_time_list = torch.cat(new_time_list, dim)

            # To sum encodes
            # new_time_list = None
            # for fn in self.embed_fns:
            #     if new_time_list is None:
            #         new_time_list = fn(a_series, t_idx)
            #     else:
            #         new_time_list += fn(a_series, t_idx)

            # print('new_time_list: ', new_time_list.squeeze())
            time_list.append(new_time_list)
        rtn = torch.cat(time_list, 2)
        return rtn


an_embed = Embedder_DCT(300, 8)


def gen_dct_on_the_fly(the_data, K=None):
    N, C, T, V, M = the_data.shape
    if K is None:
        K = T

    the_data = an_embed.embed(the_data, dim=1)
    return the_data


def dct_2_no_sum_parallel(bch_seq, K0=0, K1=None):
    bch_seq_rsp = bch_seq.view(-1, bch_seq.shape[1])
    N = bch_seq_rsp.shape[1]
    if K1 is None:
        K1 = N
    basis_list = []
    for k in range(K0, K1):
        a_basis_list = []
        for i in range(N):
            a_basis_list.append(math.cos(math.pi / N * (i + 0.5) * k))
        basis_list.append(a_basis_list)
    basis_list = torch.tensor(basis_list).to(bch_seq_rsp.device)
    bch_seq_rsp = bch_seq_rsp.unsqueeze(1).repeat(1, K1 - K0, 1)
    dot_prod = torch.einsum('abc,bc->abc', bch_seq_rsp, basis_list)
    return dot_prod.view(-1, K1 - K0)
