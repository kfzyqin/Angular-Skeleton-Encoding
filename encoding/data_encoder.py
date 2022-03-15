import torch
import math
import torch.nn.functional as functional


class DataRepeatEncoder:
    def __init__(self, rep_num):
        self.rep_num = rep_num

    def encode_data(self, a_bch, **kwargs):
        rtn = a_bch.repeat(1, self.rep_num, 1, 1, 1)
        return rtn


class DataInterpolatingEncoder:
    def __init__(self, new_length=None):
        self.new_length = new_length

    def encode_data(self, a_bch, **kwargs):
        N, C, T, V, M = a_bch.size()
        a_bch = a_bch.permute(0, 4, 1, 3, 2).contiguous().view(N*M*C, V, T)
        a_bch = functional.interpolate(a_bch, size=self.new_length, mode='linear')
        a_bch = a_bch.view(N, M, C, V, self.new_length).permute(0, 2, 4, 3, 1)
        return a_bch

class TrigonometricTemporalEncoder:
    def __init__(self, inc_type, freq_num, seq_len, is_with_orig=True):
        self.inc_func = inc_type
        self.periodic_fns = [torch.cos]
        self.is_with_orig = is_with_orig
        self.K = freq_num
        self.T = seq_len
        self.prepare_period_fns()

    def prepare_period_fns(self):
        assert self.inc_func is not None
        assert self.periodic_fns is not None

        # Get frequency values
        self.temp_freq_bands = []
        for k in range(1, self.K + 1):
            if self.inc_func == 'linear':
                a_freq = k
            elif self.inc_func == 'exp':
                a_freq = 2 ** (k - 1)
            elif self.inc_func == 'pow':
                a_freq = k ** 2
            else:
                raise NotImplementedError('Unsupported inc_func.')
            self.temp_freq_bands.append(math.pi / self.T * a_freq)

        self.temp_freq_bands = torch.tensor(self.temp_freq_bands)
        print('Temporal frequency components: ', self.temp_freq_bands)

        # Get embed functions
        self.temp_embed_fns = []
        if self.is_with_orig:
            self.temp_embed_fns.append(lambda x, frm_idx: x)

        for freq_t in self.temp_freq_bands:
            for p_fn in self.periodic_fns:
                self.temp_embed_fns.append(
                    lambda x, frm_idx, p_fn=p_fn, freq=freq_t: (x * p_fn(freq * (frm_idx + 1 / 2)))
                )  # TTE

    def encode_data(self, a_bch, dim):
        t_len_all = a_bch.shape[2]
        time_list = []
        for t_idx in range(t_len_all):
            a_series = a_bch[:, :, t_idx, :, :].unsqueeze(2)

            new_time_list = []
            for fn in self.temp_embed_fns:
                a_new_one = fn(a_series, t_idx)
                new_time_list.append(a_new_one)
            new_time_list = torch.cat(new_time_list, dim)

            time_list.append(new_time_list)
        rtn = torch.cat(time_list, 2)
        return rtn
