import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

# In theory 100 is our alpha_i_bar
# So we imagine that alpha_i_bar is decided by us
# THIS IS THE FUNCTION THAT CHANGE
# SEE THE GRAPH => IT IS THE "EXTENSION" OF THE ONE VARIABLE
def n(x, y):
    # the real function to estimate
    return (1.0 - np.exp(- (x * y))) * 100


def generate_observation(x, y, noise_std):
    return n(x, y) + np.random.normal(0, noise_std, size=n(x, y).shape)


alpha_bar = 100
n_obs = 20

bids, budgets = np.meshgrid(np.arange(start=0.0, stop=3.0, step=0.02), np.arange(start=0.0, stop=3.0, step=0.02))

X_obs = np.array([0, 0])
z_obs = np.array([0])  # corresponding (to x_obs, y_obs) observed clicks

# Increasing noise => increasing uncertainty
noise_std = 0.1

X_pred = np.array([0, 0])

# Real function printed

for i in range(0, n_obs):
    # Generate a point based on the value of the bid (chosen randomly)
    # at each time a random point is chosen
    new_bid_obs = np.random.choice(bids.ravel(), 1).reshape(-1)[0]
    new_budget_obs = np.random.choice(budgets.ravel(), 1).reshape(-1)[0]
    new_z_obs = generate_observation(new_bid_obs, new_budget_obs, noise_std)

    vec = np.array([new_bid_obs, new_budget_obs])
    X_obs = np.vstack((X_obs, vec))
    z_obs = np.append(z_obs, new_z_obs)

    X = X_obs
    Z = z_obs.ravel().reshape(-1, 1)

    theta = 1.0
    l = 1.0
    kernel = C(theta, (1e-3, 1e3)) * RBF(l, (1e-3, 1e3))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=noise_std ** 2, normalize_y=True, n_restarts_optimizer=10)

    gp.fit(X, Z)

    Z_pred, sigma = gp.predict(np.array([bids.ravel(), budgets.ravel()]).T, return_std=True)

print(f"bids size = {bids.shape}; budgets size = {budgets.shape}; Z_Pred size = {Z_pred.reshape(150, 150).shape}")


# Real function printed
z = n(bids, budgets)
ax = plt.axes(projection='3d')
ax.plot_surface(bids, budgets, z, cmap='viridis', edgecolor='green')
plt.show()

# Learned
ax = plt.axes(projection='3d')
ax.plot_surface(bids, budgets, Z_pred.reshape(150, 150), cmap='viridis', edgecolor='green')
plt.show()



