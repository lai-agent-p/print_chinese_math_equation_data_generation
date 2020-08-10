'''
utilities for neural networks(pytorch only), including
parameter loading code, nn module that can be shared.
'''
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
from os import listdir
import numpy as np
import pdb
from torch.optim.optimizer import Optimizer
from torch.distributions.normal import Normal
import math

def save_network(network,
                 save_dir,
                 model_name,
                 epoch_label,
                 curr_accy,
                 best=False,
                 pretrained_ind=None):
    '''
    save current neural network parameters
    :param network: the pytorch neural network that should be saved
                    the neural net is assumed to have optimizer as a module
                    aka there is a line: self.optimizer = nn.OptimizerName(...)
    :param save_dir: path to the checkpoint directory
    :param model_name: name of the model
    :param epoch_label: current epoch number
    :return: None
    '''
    if best:
        save_filename = 'net_%s_best.pth' % (model_name)
    else:
        save_filename = '%s_net_%s.pth' % (epoch_label, model_name)
    save_path = os.path.join(save_dir, save_filename)

    if pretrained_ind is None:
        state = {'state_dict': network.cpu().state_dict(),
                 'optimizer': network.optimizer.state_dict(),
                 'accuracy': curr_accy}
    else:
        state = {'state_dict': network.cpu().state_dict(),
                 'optimizer': network.optimizer.state_dict(),
                 'accuracy': curr_accy,
                 'pretrain_inds': torch.from_numpy(np.asarray(pretrained_ind))}

    torch.save(state, save_path)
    network.cuda()


def initialize_model(network,
                     lexicon,
                     shining_lexicon,
                     shining_model_path
                     ):
    '''
    initialize model

    :param network: crnn-network
    :param lexicon: lexicon for the model
    :param shining_lexicon: shining for lexicon
    :param shining_model_path:  path to shining's model path
    :return: network with loaded parameter, epoch, ind for words that has embedding parameter on shing model
    '''
    if lexicon is None:
        assert (shining_lexicon is None and shining_model_path is None)
        print('initialize without shining embedding')
        return network, 0, 0
    else:
        print('initialize with shining embedding')
        # load shining model state dict
        state_dicts = torch.load(shining_model_path)
        new_word_embedding_weights = []
        shining_word_embeddings = state_dicts['model']['module.bert.embeddings.word_embeddings.weight']
        shining_keys = set(shining_lexicon)
        key_sets = set(lexicon)
        print('{} are keys not in shining keys'.format(key_sets.difference(shining_keys)))
        shining_word2ind = {}
        for i in range(len(shining_lexicon)):
            shining_word2ind[shining_lexicon[i]] = i
        pretrained_inds = []
        keys = list(lexicon)
        for i, w in enumerate(keys):
            if w in shining_keys:
                pretrained_inds.append(i)
                new_word_embedding_weights.append(shining_word_embeddings[shining_word2ind[w]].type(torch.FloatTensor))
            else:
                new_word_embedding_weights.append(
                    nn.init.normal_(state_dicts['model']['module.bert.embeddings.word_embeddings.weight'][0]).type(
                        torch.FloatTensor))

        new_word_embedding_weights = torch.stack(
            new_word_embedding_weights)  # convert a list of tensors to a single tensor.
        network.decoder.embedding.weight = nn.Parameter(new_word_embedding_weights)

        return network, 0, pretrained_inds


def load_last_checkpoint(network,
                         save_dir,
                         model_name,
                         use_gpu,
                         lexicon=None,
                         shining_lexicon=None,
                         shining_model_path=None
                         ):
    '''
    load the latest checkpoint from  checkpoint folder
    :param network: the pytorch neural network (to be safe the model should be on cpu)
                    the neural net is assumed to have optimizer as a module
                    aka there is a line: self.optimizer = nn.OptimizerName(...)
    :param save_dir: where the checkpoint file sits
    :param model_name: name of the model
    :return: network with loaded parameters, iternumber of last checkpoint
    '''
    checkpoint_paths = [f for f in listdir(save_dir) if f.endswith('pth')]
    if not len(checkpoint_paths):
        print('first iteration, initialize model')
        return initialize_model(network,
                                lexicon,
                                shining_lexicon,
                                shining_model_path)

    iter_numbers = [int(f.split('_')[0]) for f in checkpoint_paths if not f[0] == 'n']
    max_iter_number = np.max(np.asarray(iter_numbers))

    network, pretrained_ind, accuracy = load_network(network,
                                                     save_dir,
                                                     model_name,
                                                     max_iter_number,
                                                     use_gpu,
                                                     lexicon=lexicon)
    if (lexicon is None):
        return network, max_iter_number, accuracy
    return network, max_iter_number, pretrained_ind, accuracy


def load_best_checkpoint(network,
                         save_dir,
                         model_name,
                         use_gpu,
                         lexicon=None,
                         shining_lexicon=None,
                         shining_model_path=None,
                         reset_optimizer=False
                         ):
    '''
    load the latest checkpoint from  checkpoint folder
    :param network: the pytorch neural network (to be safe the model should be on cpu)
                    the neural net is assumed to have optimizer as a module
                    aka there is a line: self.optimizer = nn.OptimizerName(...)
    :param save_dir: where the checkpoint file sits
    :param model_name: name of the model
    :return: network with loaded parameters, iternumber of last checkpoint
    '''
    checkpoint_paths = [f for f in listdir(save_dir) if f.endswith('pth')]
    if not len(checkpoint_paths):
        print('first iteration, initialize model')
        return initialize_model(network,
                                lexicon,
                                shining_lexicon,
                                shining_model_path)
    if 'net_%s_best.pth' % model_name in checkpoint_paths:
        max_iter_number = None
    else:
        iter_numbers = [int(f.split('_')[0]) for f in checkpoint_paths]
        max_iter_number = np.max(np.asarray(iter_numbers))

    network, pretrained_ind, accuracy = load_network(network,
                                                     save_dir,
                                                     model_name,
                                                     max_iter_number,
                                                     use_gpu,
                                                     reset_optimizer=reset_optimizer,
                                                     lexicon=lexicon)
    if (lexicon is None):
        return network, max_iter_number, accuracy
    return network, max_iter_number, pretrained_ind, accuracy


def load_network(network,
                 save_dir,
                 model_name,
                 epoch_label,
                 use_gpu,
                 reset_optimizer=False,
                 lexicon=None):
    '''
    load neural network parameters
    :param network: the pytorch neural network (to be safe the model should be on cpu)
                    the neural net is assumed to have optimizer as a module
                    aka there is a line: self.optimizer = nn.OptimizerName(...)
    :param save_dir: where the checkpoint file sits
    :param model_name: name of the model
    :param epoch_label: how many epoches the model was trained (which checkpoint you want to load)
                        or simply a label to distinguish checkpoints, in this project, it is the
                        iteration number
    :return: network with loaded parameters
    '''
    if epoch_label is None:
        save_filename = 'net_%s_best.pth' % (model_name)
    else:
        save_filename = '%s_net_%s.pth' % (epoch_label, model_name)
    save_path = os.path.join(save_dir, save_filename)
    state_dicts = torch.load(save_path)

    mask_keys = [key for key in state_dicts['state_dict'].keys() if 'bernoulli_mask' in key]
    for mask_key in mask_keys:
        del state_dicts['state_dict'][mask_key]
    network.load_state_dict(state_dicts['state_dict'], strict=False)
    if use_gpu:
        network = network.cuda()
    print('epoch label is {}'.format(epoch_label))
    if not reset_optimizer or epoch_label == 1:
        network.optimizer.load_state_dict(state_dicts['optimizer'])
    if not lexicon is None:
        pretrained_ind = state_dicts['pretrained_ind'].cpu().numpy()
    else:
        pretrained_ind = None

    if 'accuracy' in state_dicts.keys():
        accuracy = state_dicts['accuracy']
    else:
        accuracy = 0
    print('load checkpoint from {}'.format(save_filename))
    return network, pretrained_ind, accuracy


class ZoneOutLSTMCell(nn.Module):
    '''
    implementation of zoneout lstm cell with  layernorm:
    Krueger, David, et al.
    "Zoneout: Regularizing rnns by randomly preserving hidden activations."
    arXiv preprint arXiv:1606.01305 (2016).
    Basically it changes certain activations O_i to O_i-1
    implementation details are checked with official tensorflow code

    :param input_size: hidden dimension of input tensor
    :param hidden_size: hidden dimension of the rnn cell
    :param zoneout_factor_cell: zoneout factor for output c
    :param zoneout_factor_output: zoneout factor for output h
    :param activation_function: activation function , by default tanh
    '''

    def __init__(self,
                 input_size,
                 hidden_size,
                 zoneout_factor_cell=0.5,
                 zoneout_factor_output=0.05,
                 activation_function=nn.Tanh()):

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.af = activation_function

        self.factor_cell = zoneout_factor_cell
        self.factor_output = zoneout_factor_output

        self.ln_f = nn.LayerNorm(hidden_size)
        self.ln_i = nn.LayerNorm(hidden_size)
        self.ln_o = nn.LayerNorm(hidden_size)
        self.ln_g = nn.LayerNorm(hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)

        self.matrix_width = 4

        weight_ih_data = init.orthogonal_(torch.Tensor(self.input_size, self.matrix_width * self.hidden_size))
        weight_hh_data = torch.eye(self.hidden_size).repeat(1, self.matrix_width)
        combined_weights = torch.cat((weight_hh_data, weight_ih_data), 0)
        combined_weights.requires_grad = True
        self.combined_weights = torch.nn.Parameter(combined_weights)
        # This seems like a hacky way to implement zoneout, but I'm not sure what the correct way would be
        # generate zoneout map that decides which activations should be zone outed with bernoulli distribution
        self.register_buffer('cell_bernoulli_mask',
                             torch.Tensor(1).fill_(zoneout_factor_cell).expand((self.hidden_size,)))

        self.register_buffer('out_bernoulli_mask',
                             torch.Tensor(1).fill_(zoneout_factor_output).expand((self.hidden_size,)))

    def forward(self, input, state):
        """
        take input and state and do a time step for normal lstm
        :param A (batch, input_size) tensor containing input
                features.
        :param A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        :return: h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = state
        combined_inputs = torch.cat((h_0, input), 1)
        preactivations = torch.mm(combined_inputs, self.combined_weights)

        f, i, o, g = torch.split(preactivations, split_size_or_sections=self.hidden_size, dim=-1)
        f = self.ln_f(f)
        i = self.ln_i(i)
        o = self.ln_o(o)
        g = self.ln_g(g)

        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * self.af(g)

        # apply zoneout on c
        if self.factor_cell > 0:
            if self.training:
                c_mask = Variable(torch.bernoulli(self.cell_bernoulli_mask))
                c_1 = c_0 * c_mask + c_1 * (1 - c_mask)

        # normalize c
        c_1 = self.ln_c(c_1)

        # apply zoneout on h
        h_1 = torch.sigmoid(o) * self.af(c_1)
        if self.factor_output > 0:
            if self.training:
                h_mask = Variable(torch.bernoulli(self.out_bernoulli_mask))
                h_1 = h_0 * h_mask + h_1 * (1 - h_mask)

        return h_1, c_1


class FocalLoss(nn.Module):
    '''
    FocalLoss proposed by Kaiming He et al.
    a tool used to aid sample imbalance.
    :param gamma: a scalar parameter decides the extent the model
                      should ignore confident samples
    :param alpha:  a list whose shape is [number of labels]
                   decides how we should update for each label
                   alpha should be bigger for rare samples
                   (alpha is not necessarily a simplex)
    :param size_average: This parameter decides if the model averages loss
                         on batch dimension
    '''

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class StackedRNNCell(nn.Module):
    def __init__(self,
                 *rnn_cells):
        '''
        initialize the stacked rnn cell with multiple rnn cells
        :param rnn_cells: the list of rnn_cells used for the
        stacked rnn cell
        '''

        super().__init__()
        self.rnn_cells = rnn_cells

    def forward(self, input, states):
        '''
        run the stacked rnn cell.
        :param input: input to stacked rnn, the hidden size of the input
                      should match the required hidden size for first rnn cell
                      in the stacked rnn cell.
        :param states: a list of states for all cells in the stacked rnn cell
                        [h,c] * number of rnn cells in the stacked rnn cell
        :return: a list of states for all rnn cells in the stacked rnn cell
        '''
        assert len(states) == len(self.rnn_cells)
        cur_input = input
        outs = []
        for i in range(len(states)):
            h, c = self.rnn_cells[i](cur_input, states[i])
            outs.append([h, c])
            cur_input = h
        return outs


class VariationalDropoutLSTMCell(nn.Module):
    '''
    implementation of lstm cell with variational dropout
    "A Theoretically Grounded Application of Dropout in
     Recurrent Neural Networks"
    Basically it changes certain activations O_i to O_i-1
    implementation details are checked with official tensorflow code

    :param input_size: hidden dimension of input tensor
    :param hidden_size: hidden dimension of the rnn cell
    :param drop_out_rate: recurrent drop out rate
    '''

    def __init__(self,
                 input_size,
                 hidden_size,
                 drop_out_rate=0.5):

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.af = nn.Tanh()

        self.recurrent_drop_out_rate = drop_out_rate

        self.matrix_width = 4

        weight_if_data = init.orthogonal_(torch.Tensor(self.input_size + self.hidden_size, self.hidden_size))
        weight_ii_data = init.orthogonal_(torch.Tensor(self.input_size + self.hidden_size, self.hidden_size))
        weight_io_data = init.orthogonal_(torch.Tensor(self.input_size + self.hidden_size, self.hidden_size))
        weight_ig_data = init.orthogonal_(torch.Tensor(self.input_size + self.hidden_size, self.hidden_size))

        weight_if_data.requires_grad = True
        weight_ii_data.requires_grad = True
        weight_io_data.requires_grad = True
        weight_ig_data.requires_grad = True

        self.weight_if_weights = torch.nn.Parameter(weight_if_data)
        self.weight_ii_weights = torch.nn.Parameter(weight_ii_data)
        self.weight_io_weights = torch.nn.Parameter(weight_io_data)
        self.weight_ig_weights = torch.nn.Parameter(weight_ig_data)

    def resample_mask(self, batch_size):
        if self.recurrent_drop_out_rate > 0.0 and self.training:
            self.dropout_mask_if = torch.bernoulli(
                1 - self.recurrent_drop_out_rate * torch.ones(batch_size, self.input_size + self.hidden_size)).cuda()
            self.dropout_mask_ii = torch.bernoulli(
                1 - self.recurrent_drop_out_rate * torch.ones(batch_size, self.input_size + self.hidden_size)).cuda()
            self.dropout_mask_io = torch.bernoulli(
                1 - self.recurrent_drop_out_rate * torch.ones(batch_size, self.input_size + self.hidden_size)).cuda()
            self.dropout_mask_ig = torch.bernoulli(
                1 - self.recurrent_drop_out_rate * torch.ones(batch_size, self.input_size + self.hidden_size)).cuda()

    def forward(self, input, state):
        """
        take input and state and do a time step for normal lstm
        :param A (batch, input_size) tensor containing input
                features.
        :param A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        :return: h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = state
        combined_inputs = torch.cat((h_0, input), 1)
        cur_batch_size = h_0.size(0)
        if self.recurrent_drop_out_rate > 0.0 and self.training:
            f_in = combined_inputs * self.dropout_mask_if[:cur_batch_size] / (1 - self.recurrent_drop_out_rate)
            i_in = combined_inputs * self.dropout_mask_ii[:cur_batch_size] / (1 - self.recurrent_drop_out_rate)
            o_in = combined_inputs * self.dropout_mask_io[:cur_batch_size] / (1 - self.recurrent_drop_out_rate)
            g_in = combined_inputs * self.dropout_mask_ig[:cur_batch_size] / (1 - self.recurrent_drop_out_rate)
        else:
            f_in = combined_inputs
            i_in = combined_inputs
            o_in = combined_inputs
            g_in = combined_inputs

        f = torch.mm(f_in, self.weight_if_weights)
        i = torch.mm(i_in, self.weight_ii_weights)
        o = torch.mm(o_in, self.weight_io_weights)
        g = torch.mm(g_in, self.weight_ig_weights)

        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * self.af(g)
        h_1 = torch.sigmoid(o) * self.af(c_1)

        return h_1, c_1


class BatchNorm2dWithMask(nn.BatchNorm2d):
    def forward(self, x, mask):
        '''
        forward function for BatchNorm2dWithMask, the mask is only used when
        calculating mean and variance

        :param x: the input tensor, should be a 4d tensor(batch_size, hidden_dim, h', w')
        :param mask: mask for the tensor (batch_size, h', w')
        :return:
        '''
        self._check_input_dim(x)
        y = x.transpose(0, 1)  # (hidden_dim, batch_size, h', w')
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)  # (hidden_dim, batch_size*h'*w')
        squeezed_mask = mask.transpose(0, 1).view(-1)  # (batch_size*h*w)
        mu = y.sum(dim=1) / squeezed_mask.sum()
        sigma2 = (((y - mu.unsqueeze(1)) ** 2) * squeezed_mask.unsqueeze(0)).sum(dim=1) / squeezed_mask.sum()
        if self.training is not True:
            y = y - self.running_mean.view(-1, 1)
            y = y / (self.running_var.view(-1, 1) ** .5 + self.eps)
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * sigma2
            y = y - mu.view(-1, 1)
            y = y / (sigma2.view(-1, 1) ** .5 + self.eps)

        y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
        return y.view(return_shape).transpose(0, 1)


def position_encoding_init(n_position, emb_dim, height):
    ''' Init the sinusoid position encoding table '''
    # pdb.set_trace()
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # apply cos on 1st,3rd,5th...emb_dim
    if height == None:
        return torch.from_numpy(position_enc).type(torch.FloatTensor)

    # keep dim 0 for padding token position encoding zero vector
    ver_dir_dim = emb_dim // 2
    hor_dir_dim = emb_dim - emb_dim // 2

    hor_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / hor_dir_dim) for j in range(hor_dir_dim)]
        if pos != 0 else np.zeros(hor_dir_dim) for pos in range(n_position)])

    hor_enc[1:, 0::2] = np.sin(hor_enc[1:, 0::2])  # apply sin on 0th,2nd,4th...emb_dim
    hor_enc[1:, 1::2] = np.cos(hor_enc[1:, 1::2])  # apply cos on 1st,3rd,5th...emb_dim

    ver_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / ver_dir_dim) for j in range(ver_dir_dim)]
        if pos != 0 else np.zeros(ver_dir_dim) for pos in range(height)])

    ver_enc[1:, 0::2] = np.sin(ver_enc[1:, 0::2])  # apply sin on 0th,2nd,4th...emb_dim
    ver_enc[1:, 1::2] = np.cos(ver_enc[1:, 1::2])  # apply cos on 1st,3rd,5th...emb_dim

    return torch.from_numpy(hor_enc).type(torch.FloatTensor), torch.from_numpy(ver_enc).type(
        torch.FloatTensor), torch.from_numpy(position_enc).type(torch.FloatTensor)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class WeightnoiseAdadelta(Optimizer):
    """Implements Adadelta algorithm.

    It has been proposed in `ADADELTA: An Adaptive Learning Rate Method`__.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    __ https://arxiv.org/abs/1212.5701
    """

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= rho <= 1.0:
            raise ValueError("Invalid rho value: {}".format(rho))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.mu_noise_coefficient = 1e-2
        self.sigma_cofficient = 1e-2
        self.log_sigma_scale = 2048.0

        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super(WeightnoiseAdadelta, self).__init__(params, defaults)

    def run_adadelta(self, square_avg, rho, grad, eps, acc_delta, group, weight):
        square_avg.mul_(rho).addcmul_(1 - rho, grad, grad)
        std = square_avg.add(eps).sqrt_()
        delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)
        weight.add_(-group['lr'], delta)
        acc_delta.mul_(rho).addcmul_(1 - rho, delta, delta)

    def step(self, n_sample, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        weight_sum = 0
        diff_sum = 0
        n_weight = 0
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adadelta does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    init_sigma = 1.0e-6
                    state['prior_mu'] = torch.tensor([0.0023]).to(p.data.device)
                    state['prior_sigma2'] = torch.tensor([0.0046]).to(p.data.device)
                    state['step'] = 0
                    state['square_avg_mu'] = torch.zeros_like(p.data)
                    state['acc_delta_mu'] = torch.zeros_like(p.data)
                    state['square_avg_logsigma'] = torch.zeros_like(p.data)
                    state['acc_delta_logsigma'] = torch.zeros_like(p.data)
                    state['mu'] = p.data.detach().clone()
                    state['logsigma'] = torch.log(torch.ones_like(p.data) * init_sigma) * 2 / self.log_sigma_scale
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                sigma2 = torch.exp(state['logsigma'] * self.log_sigma_scale)
                rho, eps = group['rho'], group['eps']

                mu_grad = self.mu_noise_coefficient * (state['mu'] - state['prior_mu']) / (state['prior_sigma2'] * n_sample) + grad
                logsigma_grad = self.sigma_cofficient * 0.5 * (sigma2/state['prior_sigma2'] - 1.0) / (self.log_sigma_scale * n_sample) + 0.5 * grad ** 2 * self.log_sigma_scale * sigma2

                state['step'] += 1

                weight_sum += torch.sum(state['mu'])
                diff_sum += torch.sum(sigma2 + (state['mu'] - state['prior_mu']) ** 2)
                n_weight += state['mu'].data.numel()

                self.run_adadelta(state['square_avg_mu'], rho, mu_grad, eps, state['acc_delta_mu'], group, state['mu'])
                self.run_adadelta(state['square_avg_logsigma'], rho, logsigma_grad, eps, state['acc_delta_logsigma'], group,
                                  state['logsigma'])


                distr = Normal(state['mu'], torch.sqrt(torch.exp(state['logsigma'] * self.log_sigma_scale)))
                new_weight = distr.sample()
                new_weight.requires_grad = True
                p.data = new_weight

        state['prior_sigma2'] = diff_sum / n_weight
        state['prior_mu'] = weight_sum / n_weight
        return loss


class WeightNoiseAdam(Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.mu_noise_coefficient = 1e-2
        self.sigma_cofficient = 1e-2
        self.log_sigma_scale = 2048.0

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(WeightNoiseAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(WeightNoiseAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def run_adam(self, exp_avg, exp_avg_sq, step, beta1, beta2, grad, group, weight):
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

        step_size = group['lr'] / bias_correction1
        weight.addcdiv_(exp_avg, denom, value=-step_size)

    @torch.no_grad()
    def step(self, n_sample, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        weight_sum = 0
        diff_sum = 0
        n_weight = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    init_sigma = 1.0e-6
                    state['prior_mu'] = torch.tensor([0.0023]).to(p.data.device)
                    state['prior_sigma2'] = torch.tensor([0.0046]).to(p.data.device)
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['mean_exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['mean_exp_avg_sq'] = torch.zeros_like(p.data)
                    state['mean'] = p.data.detach().clone()
                    # Exponential moving average of gradient values
                    state['logsigma_exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['logsigma_exp_avg_sq'] = torch.zeros_like(p.data)
                    state['logsigma'] = torch.log(torch.ones_like(p.data) * init_sigma) * 2 / self.log_sigma_scale

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                sigma2 = torch.exp(state['logsigma'] * self.log_sigma_scale)

                mu_grad = self.mu_noise_coefficient * (state['mean'] - state['prior_mu']) / (
                            state['prior_sigma2'] * n_sample) + grad
                logsigma_grad = self.sigma_cofficient * 0.5 * (sigma2 / state['prior_sigma2'] - 1.0) / (
                            self.log_sigma_scale * n_sample) + 0.5 * grad ** 2 * self.log_sigma_scale * sigma2


                state['step'] += 1

                weight_sum += torch.sum(state['mean'])
                diff_sum += torch.sum(sigma2 + (state['mean'] - state['prior_mu']) ** 2)
                n_weight += p.data.numel()
                beta1, beta2 = group['betas']
                state['step'] += 1

                self.run_adam(state['mean_exp_avg'], state['mean_exp_avg_sq'], state['step'], beta1,
                              beta2, mu_grad, group, state['mean'])
                self.run_adam(state['logsigma_exp_avg'], state['logsigma_exp_avg_sq'], state['step'], beta1,
                              beta2, logsigma_grad, group, state['logsigma'])

                distr = Normal(state['mean'], torch.sqrt(torch.exp(state['logsigma'] * self.log_sigma_scale)))

                new_weight = distr.sample()
                new_weight.requires_grad = True
                p.data = new_weight

        state['prior_sigma2'] = diff_sum / n_weight
        state['prior_mu'] = weight_sum / n_weight
        return loss