import numpy as np

sys.path.append('../Learners/')
from Learners.Learner import Learner


class GTS_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.means = np.zeros(n_arms)
        seld.sigmas = np.ones(n_arms) * 1e3

    def pull_arm(self):
        idx = np.argmax(np.random.normal(self.means, self.sigmas))
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.means[pulled_arm] = np.mean(self.rewards_per_arm[pulled_arm])
        n_samples = lem(self.rewards_per_arm[pulled_arm])
        if n_samples > 1:
            self.sigmas[pulled_arm] = np.std(self.rewards_per_arm[pulled_arm])/n_samples
