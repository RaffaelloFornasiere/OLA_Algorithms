import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# given a bid returns the expected number of clicks
def n(x):
    return (1.0 - np.exp(-5.0 * x)) * 100


# generate an observation by adding some random noise to number of clicks
def generate_observation(x, noise_std):
    return n(x) + np.random.normal(0, noise_std, size=n(x).shape)


n_obs = 50
bids = np.linspace(0.0, 1.0, 20)  # linear space from 0 to 1 of 20 vals
x_obs = np.array([])  # served bids
y_obs = np.array([])  # corresponding (to x_obs) observed clicks

noise_std = 5.0

for i in range(0, n_obs):
    new_x_obs = np.random.choice(bids, 1)
    new_y_obs = generate_observation(new_x_obs, noise_std)

    x_obs = np.append(x_obs, new_x_obs)
    y_obs = np.append(y_obs, new_y_obs)

    # now we have to initialize gp by setting its parameters.
    # - normalize data (in our case they are already normalized)
    # - specify kernel function (in our case, square_exponential)
    # - set hyperparameters (a good choice when we don't know is to set l=1 and theta = 1)
    #
    X = np.atleast_2d(x_obs).T
    Y = y_obs.ravel()

    theta = 1.0
    l = 1.0
    kernel = C(theta, (1e-3, 1e3)) * RBF(l, (1e-3, 1e3))

    # when we fit the gp we'll find the optimal value of the hyperparameters my maximizing the marginal
    # likelihood. This operation is computed 10 times (n_restarts_optimizer) starting from different points
    gp = GaussianProcessRegressor(kernel=kernel, alpha=noise_std ** 2, normalize_y=True, n_restarts_optimizer=10)

    gp.fit(X, Y)

    x_pred = np.atleast_2d(bids).T
    y_pred, sigma = gp.predict(x_pred, return_std=True)

    plt.figure(i)
    plt.plot(x_pred, n(x_pred), 'r:', label=r'$n(x)$')
    plt.plot(X.ravel(), Y, 'ro', label=u'Observed Clicks')
    plt.plot(x_pred, y_pred, 'b-', label=u'Predicted Clicks')
    plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
             np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% conf interval')
    plt.xlabel('$x$')
    plt.ylabel('$n(x)$')
    plt.legend(loc="lower right")
    plt.show()
