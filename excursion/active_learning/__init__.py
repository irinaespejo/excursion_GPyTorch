import datetime
import torch
import gpytorch
import numpy as np
from scipy.stats import norm
from scipy.linalg import cho_solve
from excursion.utils import h_normal
from torch.distributions import normal
from excursion.utils import truncated_std_conditional, get_first_max_index


def acq(gp, testcase, x_candidate, acquisition: str, device, dtype):
    """
    Acquisition function
    Inputs:
    - model for Gaussian Process
    -  settings of the testcase
    - thresholds: aray with thresholds including infinity
    - x_candidate: candidate point in te grid
    """
    if acquisition == "PES":
        thresholds = [-np.inf] + testcase.thresholds.tolist() + [np.inf]
        info_gain = PES(gp, testcase, thresholds, x_candidate, device, dtype)
        return info_gain

    if acquisition == "MES":
        thresholds = [-np.inf] + testcase.thresholds.tolist() + [np.inf]
        info_gain = MES(gp, testcase, thresholds, x_candidate, device, dtype)
        return info_gain


def MES(gp, testcase, thresholds, x_candidate, device, dtype):

    # compute predictive posterior of Y(x) | train data
    kernel = gp.covar_module
    likelihood = gp.likelihood
    gp.eval()
    likelihood.eval()

    Y_pred_candidate = likelihood(gp(x_candidate))

    normal_candidate = torch.distributions.Normal(
        loc=Y_pred_candidate.mean, scale=Y_pred_candidate.variance ** 0.5
    )

    # entropy of S(x_candidate)
    entropy_candidate = torch.Tensor([0.0])

    for j in range(len(thresholds) - 1):
        # p(S(x)=j)
        p_j = normal_candidate.cdf(thresholds[j + 1]) - normal_candidate.cdf(
            thresholds[j]
        )

        if p_j > 0.0:
            # print(x_candidate, p_j,j)
            entropy_candidate -= torch.logsumexp(p_j, 0) * torch.logsumexp(
                torch.log(p_j), 0
            )

    return float(entropy_candidate.detach().numpy())


def PES(gp, testcase, thresholds, x_candidate, device, dtype):
    """
    Calculates information gain of choosing x_candidadate as next point to evaluate.
    Performs this calculation with the Predictive Entropy Search approximation. Roughly,
    PES(x_candidate) = int dx { H[Y(x_candidate)] - E_{S(x=j)} H[Y(x_candidate)|S(x=j)] }
    Notation: PES(x_candidate) = int dx H0 - E_Sj H1

    """

    # compute predictive posterior of Y(x) | train data
    kernel = gp.covar_module
    likelihood = gp.likelihood
    gp.eval()
    likelihood.eval()

    X_grid = (testcase.X).to(device, dtype)
    X_all = torch.cat((x_candidate, X_grid)).to(device, dtype)
    Y_pred_all = likelihood(gp(X_all))
    Y_pred_grid = torch.distributions.Normal(
        loc=Y_pred_all.mean[1:], scale=(Y_pred_all.variance[1:]) ** 0.5
    )

    # vector of expected value H1 under S(x) for each x in X_grid
    E_S_H1 = torch.zeros(len(X_grid)).to(device, dtype)

    for j in range(len(thresholds) - 1):

        # vector of sigma(Y(x_candidate)|S(x)=j) truncated
        trunc_std_j = truncated_std_conditional(
            Y_pred_all, thresholds[j], thresholds[j + 1]
        )
        H1_j = h_normal(trunc_std_j)

        # vector of p(S(x)=j)
        p_j = Y_pred_grid.cdf(thresholds[j + 1]) - Y_pred_grid.cdf(thresholds[j])
        mask = torch.where(p_j == 0.0)
        H1_j[mask] = 0.0
        E_S_H1 += p_j * H1_j  # expected value of H1 under S(x)

    # entropy of Y(x_candidate)
    H0 = h_normal(Y_pred_all.variance[0] ** 0.5)

    # info grain on the grid, vector
    info_gain = H0 - E_S_H1

    info_gain[~torch.isfinite(info_gain)] = 0.0  # just to avoid NaN

    # cumulative info gain over grid
    cumulative_info_gain = info_gain.sum()

    return cumulative_info_gain.item()
