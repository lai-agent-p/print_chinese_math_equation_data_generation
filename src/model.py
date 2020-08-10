from src.dense_net import DenseNet
from src.decoder import LstmDecoder
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import StepLR as step_scheduler
from math import pi
from src.util.nn_util import VariationalDropoutLSTMCell
from torch.distributions.normal import Normal
from src.gmm import get_mixture_coef, get_loss, sample_gaussian_2d


def is_no_decay(name):
    '''
    check if the input name corresponds to a parameter that should
    not be regularized

    :param name: name of the parameter
    :return: True if we should not apply l2 regularizer on this parameter
    '''
    return 'bias' in name  # or 'bn' in name or 'ln' in name


class LayoutModel(nn.Module):
    def __init__(self, config):
        super(LayoutModel, self).__init__()
        self.encoder = DenseNet(encoder_out_dim=config.ENCODER_OUT_DIM)
        self.decoder = LstmDecoder(encoder_dim=128 + 4, use_sample_labels=config.USE_SAMPLE_LABELS)
        self.learn_gaussian_distr = config.LEARN_GAUSSIAN_DIST
        self.do_teacher_force = config.DO_TEACHER_FORCE
        weight_decay_params = [param for name, param in self.named_parameters() if
                               param.requires_grad and not is_no_decay(name)]
        no_weight_decay_params = [param for name, param in self.named_parameters() if
                                  param.requires_grad and is_no_decay(name)]

        if config.OPTIMIZER == 'AdaDelta':
            self.optimizer = torch.optim.Adadelta([
                {'params': weight_decay_params, 'weight_decay': config.WEIGHT_DECAY},
                {'params': no_weight_decay_params}], lr=config.INIT_LEARNING_RATE)

        elif config.OPTIMIZER == 'Adam':
            self.optimizer = torch.optim.Adam([
                {'params': weight_decay_params, 'weight_decay': config.WEIGHT_DECAY},
                {'params': no_weight_decay_params}], lr=config.INIT_LEARNING_RATE)

        self.scheduler = step_scheduler(self.optimizer, config.STEP_SIZE, config.LR_DECAY_RATE)
        if self.learn_gaussian_distr:
            self.hidden2GMMparam = nn.Linear(2 * 128 + 4, config.N_PI * 6)
        else:
            self.hidden2pred = nn.Linear(2 * 128 + 4, 2)

        if self.do_teacher_force:
            self.init_h = nn.Parameter(torch.randn(1, 2 * 128 + 4), requires_grad=True).float()
            self.init_c = nn.Parameter(torch.randn(1, 2 * 128 + 4), requires_grad=True).float()
            self.out_lstm = VariationalDropoutLSTMCell(2 * 128 + 4 + 2, 2 * 128 + 4)

    def forward(self, ims, lengths, shapes, char_types, gap_means, slopes, config, labels=None):
        cuda = torch.device('cuda:0')
        n_ims = lengths
        batch_size, im_seq_len, h, w = ims.size()
        ims = ims.view(batch_size * im_seq_len, 1, h, w)
        encoded_im = self.encoder(ims)
        seq_encoded_im = encoded_im.view(batch_size, im_seq_len, -1)
        out_feature = self.decoder(seq_encoded_im, shapes, n_ims, gap_means, slopes, char_types)
        if self.do_teacher_force:
            step_h = self.init_h.expand(batch_size, -1)
            step_c = self.init_c.expand(batch_size, -1)
            self.out_lstm.resample_mask(batch_size)

            if self.training:
                _, _, label_dim = labels.size()
                if encoded_im.is_cuda:
                    lstm_out = torch.zeros(batch_size, im_seq_len, 2 * 128 + 4, device=cuda)
                else:
                    lstm_out = torch.zeros(batch_size, im_seq_len, 2 * 128 + 4)

                for t in range(max(n_ims)):
                    batch_size_t = sum([l > t for l in n_ims])
                    step_feature = out_feature[:batch_size_t, t, :]
                    step_h = step_h[:batch_size_t, :]
                    step_c = step_c[:batch_size_t, :]
                    if t == 0:
                        if encoded_im.is_cuda:
                            step_label = torch.zeros((batch_size_t, label_dim), device=cuda)
                        else:
                            step_label = torch.zeros((batch_size_t, label_dim))
                    else:
                        step_label = labels[:batch_size_t, t - 1, :]
                    step_feature = torch.cat([step_feature, step_label], dim=1)
                    step_h, step_c = self.out_lstm(step_feature, (step_h, step_c))
                    lstm_out[:batch_size_t, t, :] = step_h
                if self.learn_gaussian_distr:
                    preds = self.hidden2GMMparam(lstm_out)
                else:
                    preds = self.hidden2pred(lstm_out)
            else:
                if encoded_im.is_cuda:
                    preds = torch.zeros(batch_size, im_seq_len, 2, device=cuda)
                else:
                    preds = torch.zeros(batch_size, im_seq_len, 2)

                step_label = torch.zeros((batch_size, 2))
                for t in range(max(n_ims)):
                    batch_size_t = sum([l > t for l in n_ims])
                    step_feature = out_feature[:batch_size_t, t, :]
                    step_h = step_h[:batch_size_t, :]
                    step_c = step_c[:batch_size_t, :]
                    if t == 0:
                        if encoded_im.is_cuda:
                            step_label = torch.zeros((batch_size_t, 2), device=cuda)
                        else:
                            step_label = torch.zeros((batch_size_t, 2))
                    else:
                        step_label = step_label[:batch_size_t, :]
                    step_feature = torch.cat([step_feature, step_label], dim=1)
                    step_h, step_c = self.out_lstm(step_feature, (step_h, step_c))
                    if self.learn_gaussian_distr:
                        # TODO modify GMM inference
                        pi, mu1, mu2, sigma1, sigma2, corr = get_mixture_coef(self.hidden2GMMparam(step_h), config)
                        out_sampled_points = 0
                        for i in range(pi.size(1)):
                            sampled_points = sample_gaussian_2d(mu1[:, i], mu2[:, i], sigma1[:, i], sigma2[:, i],
                                                                corr[:, i], sqrt_temp=1.0, greedy=False)
                            out_sampled_points = pi[i] * out_sampled_points + sampled_points
                        preds[:batch_size_t, t, :] = out_sampled_points
                    else:
                        step_label = self.hidden2pred(step_h)
                        preds[:batch_size_t, t, :] = step_label
        else:
            if self.learn_gaussian_distr:
                if self.training:
                    preds = self.hidden2GMMparam(out_feature)
                else:
                    pi, mu1, mu2, sigma1, sigma2, corr = get_mixture_coef(self.hidden2GMMparam(out_feature), config)
                    out_sampled_points = 0
                    for i in range(pi.size(2)):
                        sampled_points = sample_gaussian_2d(mu1[:, :, i], mu2[:, :, i], sigma1[:, :, i], sigma2[:, :, i],
                                                            corr[:, :, i], sqrt_temp=1.0, greedy=False)
                        out_sampled_points = out_sampled_points + pi[:, :, i].unsqueeze(-1) * sampled_points
                    preds = out_sampled_points
            else:
                preds = self.hidden2pred(out_feature)
        return preds


# def gaussian_loss():


def compute_loss(predictions, gt_content, gt_length_list):
    '''
    compute ocr loss

    :param predictions(tensor): a sequence of prediction logits (batch_size, max_sequence_length, num_vocab))
    :param gt_content(tensor): ground truth token indicies(batch_size, max_sequence_length)
    :param gt_length_list(tensor): a pytorch list that contains length of all sequences (batch_size)
    :return: loss and accuracy of current iteration
    '''
    # targets = gt_content[:, 1:]
    targets = gt_content
    predictions, _, _, _ = pack_padded_sequence(predictions, (gt_length_list), batch_first=True)
    targets, _, _, _ = pack_padded_sequence(targets, (gt_length_list), batch_first=True)
    n_outs = targets.size()[0]
    loss_center = nn.SmoothL1Loss()(predictions[:, 0], targets[:, 0])
    loss_gap = nn.SmoothL1Loss()(predictions[:-1, 1], targets[:-1, 1])
    loss = loss_center * 0.5 + loss_gap * 0.5
    # loss = FocalLoss(gamma= 1.2)(predictions, targets)
    # model_out = torch.argmax(predictions, dim=1)
    accy = 0
    return loss, accy, n_outs


def gaussian_loss(label, mean, std):
    epsilan = 1e-10
    loss = ((label - mean) / std + epsilan) ** 2 + torch.log(2 * pi * std ** 2)
    loss = torch.mean(loss)
    return loss


def compute_loss_guass(predictions, gt_content, gt_length_list, config):
    '''
        compute ocr loss

        :param predictions(tensor): a sequence of prediction logits (batch_size, max_sequence_length, num_vocab))
        :param gt_content(tensor): ground truth token indicies(batch_size, max_sequence_length)
        :param gt_length_list(tensor): a pytorch list that contains length of all sequences (batch_size)
        :return: loss and accuracy of current iteration
        '''
    # targets = gt_content[:, 1:]
    targets = gt_content
    predictions, _, _, _ = pack_padded_sequence(predictions, (gt_length_list), batch_first=True)
    targets, _, _, _ = pack_padded_sequence(targets, (gt_length_list), batch_first=True)
    pi, mu1, mu2, sigma1, sigma2, corr = get_mixture_coef(predictions, config)
    n_outs = targets.size()[0]
    loss = get_loss(pi, mu1, mu2, sigma1, sigma2, corr, targets[:, 0], targets[:, 1])

    # loss = FocalLoss(gamma= 1.2)(predictions, targets)
    # model_out = torch.argmax(predictions, dim=1)
    accy = 0
    return loss, accy, n_outs


def train_step(model, sample, config):
    ims = sample.ims
    shapes = sample.shapes
    labels = sample.labels
    seq_lengths = sample.seq_lengths
    embed_indicies = sample.type_embeds
    gap_means = sample.gap_means
    slopes = sample.slopes

    if config.USE_GPU:
        ims = ims.cuda()
        shapes = shapes.cuda()
        labels = labels.cuda()
        seq_lengths = seq_lengths.cuda()
        embed_indicies = embed_indicies.cuda()
        gap_means = gap_means.cuda()
        slopes = slopes.cuda()

    if not config.USE_SAMPLE_LABELS:
        embed_indicies = None
    if config.DO_TEACHER_FORCE:
        preds = model(ims, seq_lengths, shapes, embed_indicies, gap_means, slopes, config, labels)
    else:
        preds = model(ims, seq_lengths, shapes, embed_indicies, gap_means, slopes, config)
    preds = preds[:, :]
    if config.LEARN_GAUSSIAN_DIST:
        loss, accy, _ = compute_loss_guass(preds, labels, seq_lengths, config)
    else:
        loss, accy, _ = compute_loss(preds, labels, seq_lengths)
    optimizer = model.optimizer
    model_scheduler = model.scheduler
    # with torch.autograd.set_detect_anomaly(True):
    loss.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_NORM)

    # check_gradient_norm(total_norm, optimizer, x, x_mask, y, seq_lengths, sample.im_ids, crnn_model, project_config)
    optimizer.step()
    model_scheduler.step()
    optimizer.zero_grad()
    return loss, accy, total_norm
