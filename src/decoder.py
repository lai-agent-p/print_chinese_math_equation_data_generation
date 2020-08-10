from src.util.nn_util import ZoneOutLSTMCell
from torch import nn
import torch
from torch.nn import functional as F
from src.util.nn_util import StackedRNNCell
import numpy as np
from src.util.misc_util import first_zero, run_nested_func
from src.util.nn_util import VariationalDropoutLSTMCell, position_encoding_init

def reverse_ignore_mask(features, length_list):
    '''
    helper function for implementing bi-directional lstm
    :param features: feature map output by encoder (batch_size*h, w, hidden), sorted by
                     length on first dimension
    :param length_list: a list that contains length of all rows on the feature map
    :return: reversed feature
    '''
    use_cuda = features.is_cuda
    rev_feature = torch.zeros_like(features)
    if use_cuda:
        rev_feature = rev_feature.cuda()
    for i, feature in enumerate(features):
        cur_len = int(length_list[i])
        rev_feature[i, :cur_len, :] = torch.flip(features[i, :cur_len, :], [1])
    return rev_feature

class LstmDecoder(nn.Module):
    def __init__(self, encoder_dim, decoder_dim=128, use_sample_labels=False):
        super(LstmDecoder, self).__init__()
        self.decoder_dim = decoder_dim
        self.decoder_lstm = VariationalDropoutLSTMCell(encoder_dim, decoder_dim)
        self.decoder_lstm_rev = VariationalDropoutLSTMCell(encoder_dim, decoder_dim)

        self.init_h = nn.Parameter(torch.randn(1, decoder_dim), requires_grad=True).float()
        self.init_c = nn.Parameter(torch.randn(1, decoder_dim), requires_grad=True).float()
        self.init_h_rev = nn.Parameter(torch.randn(1, decoder_dim), requires_grad=True).float()
        self.init_c_rev = nn.Parameter(torch.randn(1, decoder_dim), requires_grad=True).float()

        self.use_sample_labels = use_sample_labels
        if self.use_sample_labels:
            self.type_embedding = nn.Embedding(3, encoder_dim - 4, padding_idx=0)

    def forward(self, encoded_feature, shapes, n_ims, gap_means, slopes, char_types = None):
        batch_size, max_seq_len, _ = encoded_feature.size()
        self.decoder_lstm.resample_mask(batch_size)

        if self.use_sample_labels:
            encoded_feature += self.type_embedding(char_types.long())

        encoded_feature_rev = reverse_ignore_mask(encoded_feature, n_ims)
        shapes_rev = reverse_ignore_mask(shapes, n_ims)
        # preds = torch.zeros(batch_size, max_seq_len, 2)
        lstm_out = torch.zeros(batch_size, max_seq_len, self.decoder_dim)
        lstm_out_rev = torch.zeros(batch_size, max_seq_len, self.decoder_dim)
        if encoded_feature.is_cuda:
            # preds = preds.cuda()
            lstm_out = lstm_out.cuda()
            lstm_out_rev = lstm_out_rev.cuda()

        step_h = self.init_h.expand(batch_size, -1)
        step_c = self.init_c.expand(batch_size, -1)

        for t in range(max(n_ims)):
            batch_size_t = sum([l > t for l in n_ims])
            step_feature = encoded_feature[:batch_size_t, t, :]
            step_h = step_h[:batch_size_t, :]
            step_c = step_c[:batch_size_t, :]

            step_shape = shapes[:batch_size_t, t, :]
            step_gap_means = gap_means[:batch_size_t].unsqueeze(-1)
            step_slopes = slopes[:batch_size_t].unsqueeze(-1)

            step_feature = torch.cat([step_feature, step_shape, step_gap_means, step_slopes], dim=1)
            step_h, step_c = self.decoder_lstm(step_feature, (step_h, step_c))

            lstm_out[:batch_size_t, t, :] = step_h

        step_h_rev = self.init_h_rev.expand(batch_size, -1)
        step_c_rev = self.init_c_rev.expand(batch_size, -1)

        for t in range(max(n_ims)):
            batch_size_t = sum([l > t for l in n_ims])
            step_feature_rev = encoded_feature_rev[:batch_size_t, t, :]
            step_h_rev = step_h_rev[:batch_size_t, :]
            step_c_rev = step_c_rev[:batch_size_t, :]

            step_shape_rev = shapes_rev[:batch_size_t, t, :]
            step_gap_means = gap_means[:batch_size_t].unsqueeze(-1)
            step_slopes = slopes[:batch_size_t].unsqueeze(-1)

            step_feature_rev = torch.cat([step_feature_rev, step_shape_rev, step_gap_means, step_slopes], dim=1)
            step_h_rev, step_c_rev = self.decoder_lstm(step_feature_rev, (step_h_rev, step_c_rev))

            lstm_out_rev[:batch_size_t, t, :] = step_h_rev

        lstm_out_rev = reverse_ignore_mask(lstm_out_rev, n_ims)
        lstm_out = torch.cat([lstm_out, lstm_out_rev], dim=2)

        expanded_gap_means = gap_means.unsqueeze(-1).unsqueeze(-1).expand((-1,max(n_ims),-1))
        expanded_slopes = slopes.unsqueeze(-1).unsqueeze(-1).expand((-1,max(n_ims),-1))
        out_feature = torch.cat([lstm_out, shapes, expanded_gap_means, expanded_slopes], dim=2)
        # preds = self.linear_out(torch.cat([lstm_out, shapes, expanded_gap_means, expanded_slopes], dim=2))

        return out_feature