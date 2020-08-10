import torch
import numpy as np
from torch.distributions import MultivariateNormal

INF_MIN = 1e-8


def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    """The probability distribution fo bivariate Gussian, Eq(14) in paper."""
    norm1 = x1 - mu1
    norm2 = x2 - mu2
    s1s2 = s1 * s2

    # Eq(15)
    z = (norm1 / s1) ** 2 + (norm2 / s2) ** 2 - 2 * rho * (norm1 * norm2) / s1s2
    neg_rho = torch.clamp(1 - rho ** 2, INF_MIN, 1.0)
    result = torch.exp(-z / (2 * neg_rho))
    denom = 2 * np.pi * s1s2 * torch.sqrt(neg_rho)
    result = result / denom
    return result


def get_loss(pi, mu1, mu2, sigma1, sigma2, corr, x1_data, x2_data):
    """Loss_d(Eq16) and Loss_c(Eq17). (before dividing by L)"""
    x1_data = x1_data.unsqueeze(dim=1)
    x2_data = x2_data.unsqueeze(dim=1)

    result0 = tf_2d_normal(x1_data, x2_data, mu1, mu2, sigma1, sigma2, corr)
    # result1 is the Loss_d (without dividing by L)
    result1 = result0 * pi
    result1 = torch.sum(result1, dim=1, keepdim=True)
    epsilan = 1e-8
    loss = torch.mean(-torch.log(result1 + epsilan))  # avoid log(0)
    return loss  # Ld, Lc; before dividing by L


def get_mixture_coef(out_tensor, config):
    """Split the output and return the Mixture Density Network params."""

    # softmax all the pi's and pen states:
    mu1, mu2, sigma1, sigma2, corr, pi = torch.split(out_tensor, config.N_PI, dim = -1)
    m = torch.nn.Softmax(dim=-1)
    pi = m(pi)  # Eq(9)

    # exponentiate the sigmas and make corr between -1 and 1.
    sigma1 = torch.exp(sigma1)  # Eq(10)
    sigma2 = torch.exp(sigma2)
    corr = torch.tanh(corr)  # Eq(11)

    result = [pi, mu1, mu2, sigma1, sigma2, corr]
    return result


def sample_gaussian_2d(mu1, mu2, s1, s2, rho, sqrt_temp=1.0, greedy=False):
    mu1 = mu1.cpu().detach().numpy()
    mu2 = mu2.cpu().detach().numpy()
    s1 = s1.cpu().detach().numpy()
    s2 = s2.cpu().detach().numpy()
    rho = rho.cpu().detach().numpy()

    out_shape = list(mu1.shape) + [2]
    n_calc = np.cumprod(np.array(mu1.shape))[-1]
    out_array = np.zeros((n_calc, 2), dtype=np.float32)

    flat_mu1 = mu1.flatten()
    flat_mu2 = mu2.flatten()
    flat_s1 = s1.flatten()
    flat_s2 = s2.flatten()
    flat_rho = rho.flatten()

    for i in range(flat_mu1.shape[0]):
        if greedy:
            out_array[i][0] = flat_mu1[i]
            out_array[i][1] = flat_mu2[i]
        else:
            mean = [flat_mu1[i], flat_mu2[i]]
            flat_s1[i] *= sqrt_temp * sqrt_temp
            flat_s2[i] *= sqrt_temp * sqrt_temp
            cov = [[flat_s1[i] * flat_s1[i], flat_rho[i] * flat_s1[i] * flat_s2[i]], [flat_rho[i] * flat_s1[i] * flat_s2[i], flat_s2[i] * flat_s2[i]]]
            x = np.random.multivariate_normal(mean, cov, 1)  # sample randomly
            out_array[i][0] = x[0][0]
            out_array[i][1] = x[0][1]
    return torch.from_numpy(np.reshape(out_array, out_shape)).cuda()
