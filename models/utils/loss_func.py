import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn


def loss_fn_kd(scores, target_scores, T=2., weights=None):
    device = scores.device
    # just taking the time-step into consideration
    if len(scores.size()) > 2:
        classes_len = scores.size(-1)
        target_len = target_scores.size(-1)
        scores = scores.view(-1, classes_len)
        target_scores = target_scores.view(-1, target_len)
        assert scores.size(0) == target_scores.size(0), "the size of predicted and target failed matching"

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    n = scores.size(1)
    # if [scores] and [target scores]
    if n > target_scores.size(1):
        batch = scores.size(0)
        zeros_to_add = torch.zeros(batch, n - target_scores.size(1)).to(device)
        targets_norm = torch.cat([targets_norm, zeros_to_add], dim=1)

    # the soft label in distillation
    KD_loss_unnorm = - targets_norm * log_scores_norm

    # Sum the probability over the whole batch
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)
    KD_loss_unnorm = weighted_average(KD_loss_unnorm, weights=weights, dim=0)

    # Normalize
    KD_loss = KD_loss_unnorm * T ** 2

    return KD_loss


def loss_fn_kd_binary(scores, target_scores, T=2., weights=None):
    device = scores.device
    if len(scores.size()) > 2:
        classes_len = scores.size(-1)
        target_len = target_scores.size(-1)
        scores = scores.view(-1, classes_len)
        target_scores = target_scores.view(-1, target_len)
        assert scores.size(0) == target_scores.size(0), "the size of predicted and target failed matching"

    scores_norm = torch.sigmoid(scores/T)
    target_norm = torch.sigmoid(target_scores/T)

    n = scores.size(1)
    if n > target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n - target_scores.size(1)).to(device)
        target_norm = torch.cat([target_norm, zeros_to_add], dim=1)

    KD_loss_unnorm = -(target_norm * torch.log(scores_norm) + (1-target_norm) * torch.log(1-scores_norm))

    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)
    KD_loss_unnorm = weighted_average(KD_loss_unnorm, weights=weights, dim=0)

    KD_loss = KD_loss_unnorm * T ** 2

    return KD_loss


def softmax_cross_entropy(output, y, beta, reduction='sum'):
    if beta == 1.0:
        output_log_sms = F.log_softmax(output, dim=1)
    else:
        output_log_sms = torch.log(torch.exp(output * beta) / torch.exp(output * beta).sum(1, keepdim=True))
    return F.nll_loss(output_log_sms, y, reduction=reduction)


def TET_loss(outputs: torch.Tensor, labels, means, lamb):
    T = outputs.shape[0]
    Loss_es = 0
    # criterion = nn.CrossEntropyLoss().to(outputs.device)
    for t in range(T):
        Loss_es += F.cross_entropy(input=outputs[t], target=labels, reduction='mean')
        # Loss_es += criterion(outputs[t], labels)
    Loss_es = Loss_es / T
    if lamb != 0:
        # MMDLoss = nn.MSELoss().to(outputs.device)
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = F.mse_loss(outputs, y, reduction='mean')
        # Loss_mmd = MMDLoss(outputs, y)
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd


# calculate the weighted average of the loss over a batch
def weighted_average(tensor, weights=None, dim=0):
    if weights is None:
        mean = torch.mean(tensor, dim=dim)
    else:
        batch_size = tensor.size(dim) if len(tensor.size()) > 0 else 1
        assert len(weights) == batch_size
        norm_weights = torch.tensor([weight for weight in weights]).to(tensor.device)
        mean = torch.mean(norm_weights*tensor, dim=dim)
    return mean


def to_one_hot(y, classes, device=None):
    if type(y) == torch.Tensor:
        device = y.device
        y = y.cpu()
    c = np.zeros(shape=[len(y), classes], dtype=np.float32)
    c[range(len(y)), y] = 1.
    c = torch.from_numpy(c)
    return c if device is None else c.to(device)


#######################################################
# Calculate log-likelihood for various distributions ##
#######################################################
def log_Normal_standard(x, mean=0, average=False, dim=None):
    log_normal = -0.5 * torch.pow(x - mean, 2)
    if dim is not None and dim == -1:
        log_normal = log_normal.view(log_normal.size(0), -1)
        dim = 1
    if average:
        return torch.mean(log_normal, dim) if dim is not None else torch.mean(log_normal)
    else:
        return torch.sum(log_normal, dim) if dim is not None else torch.sum(log_normal)


def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    # the differen between this and the former is that the above loss set variance to default 1
    log_normal = - 0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))
    if dim is not None and dim == -1:
        log_normal = log_normal.view(log_normal.size(0), -1)
        dim = 1
    if average:
        return torch.mean(log_normal, dim) if dim is not None else torch.mean(log_normal)
    else:
        return torch.sum(log_normal, dim) if dim is not None else torch.sum(log_normal)


# Calculate the log-likehood for the Bernoulli distribution
def log_Bernoulli(x, mean, average=False, dim=None):
    probs = torch.clamp(mean, min=1e-5, max=1. - 1e-5)
    log_bernoulli = x * torch.log(probs) + (1. - x) * torch.log(1. - probs)
    if dim is not None and dim == -1:
        log_bernoulli = log_bernoulli.view(log_bernoulli.shape[0], -1)
        dim = 1
    if average:
        return torch.mean(log_bernoulli, dim) if dim is not None else torch.mean(log_bernoulli)
    else:
        return torch.sum(log_bernoulli, dim) if dim is not None else torch.sum(log_bernoulli)


