import torch
import numpy as np
# Define true functions


def function_1(X: torch.Tensor) -> torch.Tensor:
    """ Returns a torch tensor where the ith-element is the i-th true function evaluated at x"""

    f1 = 10 - 10 * (torch.tanh(X * 2) + 0.15 * torch.sin(X * 15))
    return f1


def function_2(X: torch.Tensor) -> torch.Tensor:
    """ Returns a torch tensor where the ith-element is the i-th true function evaluated at x"""

    f2 = X + 2.5 * torch.sin(X * 3)
    return f2


def function_3(X: torch.Tensor) -> torch.Tensor:
    """ Returns a torch tensor where the ith-element is the i-th true function evaluated at x"""

    f3 = 2 * X - 1
    return f3


def function_4(X: torch.Tensor) -> torch.Tensor:
    """ Returns a torch tensor where the ith-element is the i-th true function evaluated at x"""

    f4 = X
    return f4

def function_5(X):
    f = 30
    return 11-10*(torch.tanh((X-0.3)*3) + 0.15*torch.sin(X*f))


true_functions = [function_5]

# Define threshold list
thresholds = torch.Tensor([0.7])

# Define grid for acquisition function
n_dims = 1

## rangedef[i] = [lower_i, upper_i, n_i] for i in n_dims
rangedef_1 = [0, 1, 100]
rangedef = np.array([rangedef_1])

grid_1 = torch.linspace(
    start=rangedef_1[0], end=rangedef_1[1], steps=rangedef_1[2], dtype=torch.double
)
X = grid_1.view(-1, 1)

## Define grid for plotting, with same format as above, default same as X
plot_X = X


true_functions = [function_5]
# thresholds = [0.7]
# bounding_box = [[0, 1]]
# ndim = 1
# grid_step_size = [100]
#
