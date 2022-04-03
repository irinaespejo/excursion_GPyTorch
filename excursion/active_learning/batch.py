import torch
import gpytorch
import os
import itertools
from excursion.models import ExactGP_RBF, GridGPRegression_RBF
    
from copy import deepcopy



def init_gp(testcase, algorithmopts, ninit, device):
    likelihood_type = algorithmopts["likelihood"]["type"]
    modelopts = algorithmopts["model"]["type"]
    kernelopts = algorithmopts["model"]["kernel"]
    prioropts = algorithmopts["model"]["prior"]

    n_dims = testcase.n_dims
    n_funcs = len(testcase.true_functions)
    epsilon = float(algorithmopts["likelihood"]["epsilon"])
    dtype = torch.float64

    #
    # TRAIN DATA
    #
    X_grid = torch.Tensor(testcase.X_plot).to(device, dtype)
    init_type = algorithmopts["init_type"]
    noise_dist = MultivariateNormal(torch.zeros(ninit), torch.eye(ninit))

    if init_type == "random":
        indexs = np.random.choice(range(len(X_grid)), size=ninit, replace=False)
        X_init = X_grid[indexs].to(device, dtype)
        noises = epsilon * noise_dist.sample(torch.Size([])).to(device, dtype)
        y_init = [func(X_init) for func in testcase.true_functions]
        # y_init = [ func(X_init)[0].to(device, dtype) + noises for func in testcase.true_functions ]
    elif init_type == "worstcase":
        X_init = [X_grid[0]]
        X_init = torch.Tensor(X_init).to(device, dtype)
        noises = epsilon * noise_dist.sample(torch.Size([])).to(device, dtype)
        y_init = testcase.true_functions[0](X_init).to(device, dtype) + noises
    elif init_type == "custom":
        raise NotImplementedError("Not implemented yet")
    else:
        raise RuntimeError("No init data specification found")

    #
    # LIKELIHOOD
    #
    if likelihood_type == "GaussianLikelihood":
        if epsilon > 0.0:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise=torch.tensor([epsilon])
            ).to(device, dtype)
        elif epsilon == 0.0:
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=torch.tensor([epsilon])
            ).to(device, dtype)

    else:
        raise RuntimeError("unknown likelihood")

    #
    # GAUSSIAN PROCESS
    #
    models = []
    if modelopts == "ExactGP" and kernelopts == "RBF":
        for i in range(n_funcs):
            model = ExactGP_RBF(X_init, y_init[i], likelihood, prioropts).to(device)
            models.append(model)
    elif modelopts == "GridGP" and kernelopts == "RBF":
        grid_bounds = testcase.rangedef[:, :-1]
        grid_n = testcase.rangedef[:, -1]

        grid = torch.zeros(int(np.max(grid_n)), len(grid_bounds), dtype=torch.double)

        for i in range(len(grid_bounds)):
            a = torch.linspace(
                grid_bounds[i][0], grid_bounds[i][1], int(grid_n[i]), dtype=torch.double
            )

            grid[:, i] = torch.linspace(
                grid_bounds[i][0], grid_bounds[i][1], int(grid_n[i]), dtype=torch.double
            )

        for i in range(n_funcs):
            model = GridGPRegression_RBF(
                grid, X_init, y_init[i], likelihood, prioropts
            ).to(device)
            models.append(model)

    else:
        raise RuntimeError("unknown gpytorch model")

    # fit
    print("X_init ", X_init)
    print("y_init ", y_init)

    for model in models:
        model.train()
        likelihood.train()
        excursion.fit_hyperparams(model, likelihood)

    return models, likelihood


def get_gp(X, y, likelihood, algorithmopts, testcase, device):
    modelopts = algorithmopts["model"]["type"]
    kernelopts = algorithmopts["model"]["kernel"]
    prioropts = algorithmopts["model"]["prior"]

    #
    # GAUSSIAN PROCESS
    #

    # to
    X = X.to(device)
    y = y.to(device)

    if modelopts == "ExactGP" and kernelopts == "RBF":
        model = ExactGP_RBF(X, y, likelihood, prioropts).to(device)
    elif modelopts == "GridGP" and kernelopts == "RBF":
        grid_bounds = testcase.rangedef[:, :-1]
        grid_n = testcase.rangedef[:, -1]

        grid = torch.zeros(int(np.max(grid_n)), len(grid_bounds), dtype=torch.double)

        for i in range(len(grid_bounds)):
            a = torch.linspace(
                grid_bounds[i][0], grid_bounds[i][1], int(grid_n[i]), dtype=torch.double
            )

            grid[:, i] = torch.linspace(
                grid_bounds[i][0], grid_bounds[i][1], int(grid_n[i]), dtype=torch.double
            )

        model = GridGPRegression_RBF(grid, X, y, likelihood, prioropts).to(device)

    else:
        raise RuntimeError("unknown gpytorch model")

    # fit
    model.train()
    likelihood.train()
    fit_hyperparams(model, likelihood)

    return model
    



def fit_hyperparams(gp, likelihood, optimizer: str = "Adam"):
    training_iter = 100
    X_train = gp.train_inputs[0]
    y_train = gp.train_targets

    if optimizer == "LBFGS":
        optimizer = torch.optim.LBFGS(
            [
                {"params": gp.parameters()},
            ],  # Includes GaussianLikelihood parameters
            lr=0.1,
            line_search_fn=None,
        )

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

        def closure():
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from gp
            output = gp(X_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.backward()
            # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f outputscale: %.3f  noise: %.3f' % (
            # i + 1, training_iter, loss.item(),
            # gp.covar_module.base_kernel.lengthscale.item(),
            # gp.covar_module. outputscale.item(),
            # gp.likelihood.noise.item()
            # ))
            return loss

    if optimizer == "Adam":

        optimizer = torch.optim.Adam(
            [
                {"params": gp.parameters()},
            ],  # Includes GaussianLikelihood parameters
            lr=0.1,
            eps=10e-6,
        )

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from gp
            output = gp(X_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.sum().backward(retain_graph=True)
            optimizer.step()












class batchGrid(object):
    """
    A class to represent the underlying grid with useful features for batch point selection
    ...
    Attributes
    -----------
    batch_types : dict()
    grid : torch.Tensor
        the acquisition values for each point in the grid
    device : str
        device to choose grom gou or cpu
    picked_indexs list
        list to keep track of the indices already selected for query or the batch
    _ndims :  int
        dimension of the grid
    Methods
    -------
    pop(index)
        Removes index from to avoid being picked again
    update(acq_value_grid)
        Actualize the elements of the acquisition values for the same grid
    
    
    """

    def __init__(self, acq_values_of_grid, device, dtype, n_dims):
        self.grid = torch.as_tensor(acq_values_of_grid, device=device, dtype=dtype)
        self.batch_types = {
            "Naive": self.get_naive_batch,
            "KB": self.get_kb_batch,
            "Distanced": self.get_distanced_batch,
            # "Cluster": self.get_cluster_batch,
        }
        self.picked_indexs = []
        self._n_dims = n_dims
        self.device = device
        self.dtype = dtype

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        args = [a._t if hasattr(a, "_t") else a for a in args]
        ret = func(*args, **kwargs)
        return batchgrid(ret, kwargs["device"], kwargs["dtype"])

    def pop(self, index):
        self.grid[index] = torch.Tensor([(-1.0) * float("Inf")])

    def update(self, acq_values_of_grid, device, dtype):
        self.grid = torch.as_tensor(acq_values_of_grid, device=device, dtype=dtype)

    def get_first_max_index(self, gp, testcase, device, dtype):
        X_train = gp.train_inputs[0].to(device, dtype)

        new_index = torch.argmax(self.grid)
        new_x = testcase.X.to(device, dtype)[new_index]

        # if the index is not already picked nor in the training set
        # accept it ans remove from future picks
        if (new_index not in self.picked_indexs) and (
            new_x.tolist() not in X_train.tolist()
        ):
            self.pop(new_index)
            self.picked_indexs.append(new_index.item())
            return new_index.item()

        else:
            self.pop(new_index)
            return self.get_first_max_index(gp, testcase, device, dtype)

    def get_naive_batch(self, gp, testcase, batchsize, device, dtype, **kwargs):
        new_indexs = []

        while len(new_indexs) < batchsize:
            max_index = self.get_first_max_index(gp, testcase, device, dtype)
            if max_index not in new_indexs:
                new_indexs.append(max_index)
                self.pop(max_index)

            else:
                self.pop(max_index)
                max_index = self.get_first_max_index(gp, testcase, device, dtype)

        return new_indexs

    def get_kb_batch(self, gp, testcase, batchsize, device, dtype, **kwargs):
        X_train = gp.train_inputs[0].to(device, dtype)
        new_indexs = []
        fake_x_list = torch.Tensor([]).to(device, dtype)
        fake_y_list = torch.Tensor([]).to(device, dtype)

        likelihood = kwargs["likelihood"]
        algorithmopts = kwargs["algorithmopts"]
        excursion_estimator = kwargs["excursion_estimator"]
        gp_fake = deepcopy(gp)

        while len(new_indexs) < batchsize:
            max_index = self.get_first_max_index(gp, testcase, device, dtype)

            if max_index not in new_indexs:
                new_indexs.append(max_index)
                self.pop(max_index)
                fake_x = testcase.X.to(device, dtype)[max_index].reshape(1, -1)
                fake_x_list = torch.cat((fake_x_list, fake_x), 0)

                gp_fake.eval()
                likelihood.eval()
                fake_y = likelihood(gp_fake(fake_x,  model=kwargs['model_multitask'])).mean
                fake_y_list = torch.cat((fake_y_list, fake_y), 0)

                # print('******* train_targets', gp_fake.train_targets.dim(), gp_fake.train_targets)
                # print('******* model_batch_sample ', len(gp_fake.train_inputs[0].shape[:-2]))
                with gpytorch.settings.cholesky_jitter(1e-1):
                    gp_fake = gp_fake.get_fantasy_model(
                        fake_x_list, fake_y_list, noise=likelihood.noise, model=kwargs['model_multitask']
                    )

                # gp_fake = self.update_fake_posterior(
                #    testcase,
                #    algorithmopts,
                #    gp_fake,
                #    likelihood,
                #    fake_x_list,
                #    fake_y_list,
                # )

                new_acq_values = excursion_estimator.get_acq_values(gp_fake, testcase, model_multitask=kwargs['model_multitask'])
                self.update(new_acq_values, device, dtype)

            else:
                self.pop(max_index)
                max_index = self.get_first_max_index(gp_fake, testcase, device, dtype)

        return new_indexs

    def update_fake_posterior(
        self,
        testcase,
        algorithmopts,
        model_fake,
        likelihood,
        list_fake_xs,
        list_fake_ys,
    ):
        with torch.autograd.set_detect_anomaly(True):

            if self._n_dims == 1:
                # calculate new fake training data
                inputs = torch.cat(
                    (model_fake.train_inputs[0], list_fake_xs), 0
                ).flatten()
                targets = torch.cat(
                    (model_fake.train_targets.flatten(), list_fake_ys.flatten()), dim=0
                ).flatten()

            else:
                inputs = torch.cat((model_fake.train_inputs[0], list_xs), 0)
                targets = torch.cat(
                    (model_fake.train_targets, list_fake_ys), 0
                ).flatten()

            model_fake.set_train_data(inputs=inputs, targets=targets, strict=False)
            model_fake = get_gp(
                inputs, targets, likelihood, algorithmopts, testcase, self.device
            )

            likelihood.train()
            model_fake.train()
            fit_hyperparams(model_fake, likelihood)

        return model_fake

    def euclidean_distance_idxs(self, array_idxs, point_idx, testcase):
        array = testcase.X[array_idxs]  
        point = testcase.X[point_idx] #vector
        d = array - point
        d = torch.sqrt(torch.sum(d**2)) #vector
        d = torch.min(d).item() #USE DIST
        if(array_idxs == []):
            return 1e8
        else:
            return d #returns a scalar

    def get_distanced_batch(self, gp, testcase, batchsize, device, dtype, **kwargs):
        new_indexs = []
        #c times the minimum grid step of separation between selected points in batch
        c = 75 #has to be > 1
        step = min((testcase.rangedef[:,1] - testcase.rangedef[:,0])/testcase.rangedef[:,-1])
        distance = c * step

        while len(new_indexs) < batchsize:
            max_index = self.get_first_max_index(gp, testcase, device, dtype)
            if max_index not in new_indexs:
                if self.euclidean_distance_idxs(new_indexs, max_index, testcase)  >= distance:
                    new_indexs.append(max_index)
                    self.pop(max_index)

            else:
                self.pop(max_index)
                max_index = self.get_first_max_index(gp, testcase, device, dtype)

        return new_indexs