import numpy as np
from scipy.optimize import linear_sum_assignment

# isn't this the hungarian algorithm?
probs = np.array([[1/4, 1, 1/4], [1/2, 1/4, 1/4], [1/4, 1/4, 1]])
costs = -probs
print(linear_sum_assignment(costs))
